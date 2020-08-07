# Copyright 2020 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Lint as: python3
"""Tests for tensorflow_lite_support.custom_ops.ngrams."""

import timeit

from absl import logging
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow.lite.python import interpreter as interpreter_wrapper  # pylint: disable=g-direct-tensorflow-import
from tensorflow_lite_support.custom_ops.python import tflite_text_api

TEST_CASES = [
    [['this', 'is', 'a', 'test']],
    [['one']],
    [['two', 'tokens'], ['a', 'b']],
    [['has', 'three', 'tokens'], ['a', 'b', 'c'], ['0', '1', '2']],
    [['a', 'ragged', 'tensor'], ['a'], ['0', '1']],
    [[['a', 'multidimensional', 'test', 'case'], ['a', 'b', 'c', 'd', 'e']],
     [['0', '1', '2', '3', '4', '5']]],
]


INVOKES_FOR_SINGLE_OP_BENCHMARK = 1000
INVOKES_FOR_FLEX_DELEGATE_BENCHMARK = 100


class NgramsTest(tf.test.TestCase, parameterized.TestCase):

  _models = {}

  def _make_model(self, rank, width, ragged_tensor=False, flex=False):
    key = (rank, width, ragged_tensor, flex)
    if key in self._models:
      return self._models[key]

    ngrams = tf_text.ngrams if flex else tflite_text_api.ngrams

    if ragged_tensor:
      input_signature = [tf.TensorSpec(shape=[None], dtype=tf.string)]
      rs = rank - 1
      input_signature += [tf.TensorSpec(shape=[None], dtype=tf.int64)] * rs

      class Model(tf.Module):

        @tf.function(input_signature=input_signature)
        def __call__(self, values, *args):
          row_splits = list(args)
          row_splits.reverse()
          input_tensor = tf.RaggedTensor.from_nested_row_splits(
              flat_values=values, nested_row_splits=tuple(row_splits))
          output_tensor = ngrams(
              input_tensor, width, reduction_type=tf_text.Reduction.STRING_JOIN)
          output = [output_tensor.flat_values]
          output.extend(list(output_tensor.nested_row_splits))
          output.reverse()
          return tuple(output)

      tf.saved_model.save(Model(), self.get_temp_dir())
    else:
      shape = [None] * rank

      class Model(tf.Module):

        @tf.function(
            input_signature=[tf.TensorSpec(shape=shape, dtype=tf.string)])
        def __call__(self, input_tensor):
          return ngrams(
              input_tensor, width, reduction_type=tf_text.Reduction.STRING_JOIN)

      tf.saved_model.save(Model(), self.get_temp_dir())

    converter = tf.lite.TFLiteConverter.from_saved_model(self.get_temp_dir())
    converter.inference_type = tf.float32
    converter.inference_input_type = tf.float32
    converter.allow_custom_ops = not flex
    if flex:
      converter.target_spec.supported_ops = [
          tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
      ]
    model = converter.convert()
    self._models[key] = model
    return model

  @parameterized.parameters([t] for t in TEST_CASES)
  def test_width_2_tensor_equivalence(self, test_case):
    input_tensor = tf.ragged.constant(test_case).to_tensor()
    tf_output = tf_text.ngrams(
        input_tensor, 2, reduction_type=tf_text.Reduction.STRING_JOIN)

    rank = input_tensor.shape.rank
    model = self._make_model(rank, 2, ragged_tensor=False, flex=False)
    interpreter = interpreter_wrapper.InterpreterWithCustomOps(
        model_content=model, custom_op_registerers=['AddNgramsCustomOp'])
    interpreter.resize_tensor_input(0, input_tensor.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'],
                           input_tensor.numpy())
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(
        interpreter.get_output_details()[0]['index'])

    self.assertEqual(tf_output.numpy().tolist(), tflite_output.tolist())

  @parameterized.parameters([t] for t in TEST_CASES)
  def test_width_3_tensor_equivalence(self, test_case):
    input_tensor = tf.ragged.constant(test_case).to_tensor()
    tf_output = tf_text.ngrams(
        input_tensor, 3, reduction_type=tf_text.Reduction.STRING_JOIN)

    rank = input_tensor.shape.rank
    model = self._make_model(rank, 3, ragged_tensor=False, flex=False)
    interpreter = interpreter_wrapper.InterpreterWithCustomOps(
        model_content=model, custom_op_registerers=['AddNgramsCustomOp'])
    interpreter.resize_tensor_input(0, input_tensor.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'],
                           input_tensor.numpy())
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(
        interpreter.get_output_details()[0]['index'])
    self.assertEqual(tf_output.numpy().tolist(), tflite_output.tolist())

  @parameterized.parameters([t] for t in TEST_CASES)
  def test_width_2_ragged_tensor_equivalence(self, test_case):
    input_tensor = tf.ragged.constant(test_case)
    tf_output = tf_text.ngrams(
        input_tensor, 2, reduction_type=tf_text.Reduction.STRING_JOIN)

    rank = input_tensor.shape.rank
    model = self._make_model(rank, 2, ragged_tensor=True, flex=False)
    interpreter = interpreter_wrapper.InterpreterWithCustomOps(
        model_content=model, custom_op_registerers=['AddNgramsCustomOp'])
    interpreter.resize_tensor_input(0, input_tensor.flat_values.shape)
    for r in range(rank - 1):
      interpreter.resize_tensor_input(r + 1,
                                      input_tensor.nested_row_splits[r].shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'],
                           input_tensor.flat_values.numpy())
    for r in range(rank - 1):
      interpreter.set_tensor(interpreter.get_input_details()[r + 1]['index'],
                             input_tensor.nested_row_splits[r].numpy())
    interpreter.invoke()
    tflite_output_values = interpreter.get_tensor(
        interpreter.get_output_details()[0]['index'])
    self.assertEqual(tf_output.flat_values.numpy().tolist(),
                     tflite_output_values.tolist())
    for i in range(rank - 1):
      tflite_output_cur_row_splits = interpreter.get_tensor(
          interpreter.get_output_details()[i + 1]['index'])
      self.assertEqual(tf_output.nested_row_splits[i].numpy().tolist(),
                       tflite_output_cur_row_splits.tolist())

  @parameterized.parameters([t] for t in TEST_CASES)
  def test_width_3_ragged_tensor_equivalence(self, test_case):
    input_tensor = tf.ragged.constant(test_case)
    tf_output = tf_text.ngrams(
        input_tensor, 3, reduction_type=tf_text.Reduction.STRING_JOIN)

    rank = input_tensor.shape.rank
    model = self._make_model(rank, 3, ragged_tensor=True, flex=False)
    interpreter = interpreter_wrapper.InterpreterWithCustomOps(
        model_content=model, custom_op_registerers=['AddNgramsCustomOp'])
    interpreter.resize_tensor_input(0, input_tensor.flat_values.shape)
    for r in range(rank - 1):
      interpreter.resize_tensor_input(r + 1,
                                      input_tensor.nested_row_splits[r].shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'],
                           input_tensor.flat_values.numpy())
    for r in range(rank - 1):
      interpreter.set_tensor(interpreter.get_input_details()[r + 1]['index'],
                             input_tensor.nested_row_splits[r].numpy())
    interpreter.invoke()
    tflite_output_values = interpreter.get_tensor(
        interpreter.get_output_details()[0]['index'])
    self.assertEqual(tf_output.flat_values.numpy().tolist(),
                     tflite_output_values.tolist())
    for i in range(rank - 1):
      tflite_output_cur_row_splits = interpreter.get_tensor(
          interpreter.get_output_details()[i + 1]['index'])
      self.assertEqual(tf_output.nested_row_splits[i].numpy().tolist(),
                       tflite_output_cur_row_splits.tolist())

  def test_latency(self):
    latency_op = 0.0
    for test_case in TEST_CASES:
      input_tensor = tf.ragged.constant(test_case)

      rank = input_tensor.shape.rank
      model = self._make_model(rank, 3, ragged_tensor=True, flex=False)
      interpreter = interpreter_wrapper.InterpreterWithCustomOps(
          model_content=model, custom_op_registerers=['AddNgramsCustomOp'])
      interpreter.resize_tensor_input(0, input_tensor.flat_values.shape)
      for r in range(rank - 1):
        interpreter.resize_tensor_input(r + 1,
                                        input_tensor.nested_row_splits[r].shape)
      interpreter.allocate_tensors()
      interpreter.set_tensor(interpreter.get_input_details()[0]['index'],
                             input_tensor.flat_values.numpy())
      for r in range(rank - 1):
        interpreter.set_tensor(interpreter.get_input_details()[r + 1]['index'],
                               input_tensor.nested_row_splits[r].numpy())
      start_time = timeit.default_timer()
      for _ in range(INVOKES_FOR_SINGLE_OP_BENCHMARK):
        interpreter.invoke()
      latency_op = latency_op + timeit.default_timer() - start_time
    latency_op = latency_op / (
        INVOKES_FOR_SINGLE_OP_BENCHMARK * len(TEST_CASES))

    latency_flex = 0.0
    for test_case in TEST_CASES:
      input_tensor = tf.ragged.constant(test_case)

      rank = input_tensor.shape.rank
      model = self._make_model(rank, 3, ragged_tensor=True, flex=True)
      interpreter = interpreter_wrapper.Interpreter(model_content=model)
      interpreter.resize_tensor_input(0, input_tensor.flat_values.shape)
      for r in range(rank - 1):
        interpreter.resize_tensor_input(r + 1,
                                        input_tensor.nested_row_splits[r].shape)
      interpreter.allocate_tensors()
      interpreter.set_tensor(interpreter.get_input_details()[0]['index'],
                             input_tensor.flat_values.numpy())
      for r in range(rank - 1):
        interpreter.set_tensor(interpreter.get_input_details()[r + 1]['index'],
                               input_tensor.nested_row_splits[r].numpy())
      start_time = timeit.default_timer()
      for _ in range(INVOKES_FOR_FLEX_DELEGATE_BENCHMARK):
        interpreter.invoke()
      latency_flex = latency_flex + timeit.default_timer() - start_time
    latency_flex = latency_flex / (
        INVOKES_FOR_FLEX_DELEGATE_BENCHMARK * len(TEST_CASES))

    logging.info('Latency (single op): %fms', latency_op * 1000.0)
    logging.info('Latency (flex delegate): %fms', latency_flex * 1000.0)


if __name__ == '__main__':
  tf.test.main()
