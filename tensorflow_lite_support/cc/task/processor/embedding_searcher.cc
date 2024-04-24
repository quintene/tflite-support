/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_lite_support/cc/task/processor/embedding_searcher.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <vector>
#include <fstream> // Include the necessary header file

#include "tensorflow_lite_support/scann_ondevice/cc/core/partitioner.h"
#include "tensorflow_lite_support/scann_ondevice/cc/core/processor.h"
#include "tensorflow_lite_support/scann_ondevice/cc/core/searcher.h"
#include "tensorflow_lite_support/scann_ondevice/cc/core/indexer.h"
#include "tensorflow_lite_support/scann_ondevice/cc/core/serialized_searcher.pb.h"
#include "tensorflow_lite_support/scann_ondevice/cc/core/top_n_amortized_constant.h"
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "tensorflow_lite_support/cc/common.h"
#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/core/external_file_handler.h"
#include "tensorflow_lite_support/cc/task/processor/proto/embedding.pb.h"
#include "tensorflow_lite_support/cc/task/processor/proto/search_options.pb.h"
#include "tensorflow_lite_support/cc/task/processor/proto/search_result.pb.h"
#include "tensorflow_lite_support/scann_ondevice/cc/index.h"
#include "tensorflow_lite_support/scann_ondevice/cc/index_builder.h"
#include "tensorflow_lite_support/scann_ondevice/proto/index_config.pb.h"


//added for SUPPORT_ASSERT_OKANDASSIGN
//#include "tensorflow_lite_support/cc/port/status_matchers.h"
//#include "tensorflow_lite_support/cc/port/gmock.h"
//#include "tensorflow_lite_support/cc/port/gtest.h"

namespace tflite {
namespace task {
namespace processor {

namespace {

constexpr int kNoNeighborId = -1;

using ::tflite::scann_ondevice::core::AsymmetricHashFindNeighbors;
using ::tflite::scann_ondevice::core::DistanceMeasure;
using ::tflite::scann_ondevice::core::FloatFindNeighbors;
using ::tflite::scann_ondevice::core::QueryInfo;
using ::tflite::scann_ondevice::core::ScannOnDeviceConfig;
using ::tflite::scann_ondevice::core::TopN;
using ::tflite::scann_ondevice::Index;
using ::tflite::scann_ondevice::IndexConfig;
using ::tflite::support::CreateStatusWithPayload;
using ::tflite::support::StatusOr;
using ::tflite::support::TfLiteSupportStatus;
using ::tflite::task::core::ExternalFileHandler;
using ::tflite::task::processor::Embedding;

using Matrix8u =
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

absl::Status SanityCheckOptions(const SearchOptions& options) {
  if (options.max_results() < 1) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("SearchOptions.max_results must be > 0, found %d.",
                        options.max_results()),
        TfLiteSupportStatus::kInvalidArgumentError);
  }
  return absl::OkStatus();
}

absl::Status SetContents(absl::string_view file_name,
                         absl::string_view content) {
  FILE* fp = fopen(file_name.data(), "w");
  if (fp == NULL) {
    return absl::InternalError(
        absl::StrFormat("Can't open file: %s", file_name));
  }

  fwrite(content.data(), sizeof(char), content.size(), fp);
  size_t write_error = ferror(fp);
  if (fclose(fp) != 0 || write_error) {
    return absl::InternalError(
        absl::StrFormat("Error while writing file: %s. Error message: %s",
                        file_name, strerror(write_error)));
  }
  return absl::OkStatus();
}

absl::Status SanityCheckIndexConfig(const IndexConfig& config) {
  switch (config.embedding_type()) {
    case IndexConfig::UNSPECIFIED:
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "Invalid IndexConfig: embedding_type must not be left UNSPECIFIED.",
          TfLiteSupportStatus::kInvalidArgumentError);
    case IndexConfig::FLOAT:
      if (config.scann_config().has_indexer()) {
        return CreateStatusWithPayload(
            absl::StatusCode::kInvalidArgument,
            "Invalid IndexConfig: embedding_type is set to FLOAT but ScaNN "
            "config specifies a product quantization codebook.",
            TfLiteSupportStatus::kInvalidArgumentError);
      }
      break;
    case IndexConfig::UINT8:
      if (!config.scann_config().has_indexer()) {
        return CreateStatusWithPayload(
            absl::StatusCode::kInvalidArgument,
            "Invalid IndexConfig: embedding_type is set to UINT8 but ScaNN "
            "config doesn't specify a product quantization codebook.",
            TfLiteSupportStatus::kInvalidArgumentError);
      }
      break;
    default:
      return CreateStatusWithPayload(
          absl::StatusCode::kInternal,
          "Invalid IndexConfig: unexpected value for embedding_type.",
          TfLiteSupportStatus::kError);
  }
  return absl::OkStatus();
}

absl::StatusOr<DistanceMeasure> GetDistanceMeasure(
    const ScannOnDeviceConfig& config) {
  DistanceMeasure measure = config.query_distance();
  if (measure == tflite::scann_ondevice::core::UNSPECIFIED) {
    if (config.has_indexer() && config.indexer().has_asymmetric_hashing()) {
      measure = config.indexer().asymmetric_hashing().query_distance();
    } else if (config.has_partitioner()) {
      measure = config.partitioner().query_distance();
    } else {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "ScaNN config does not provide mandatory DistanceMeasure.",
          TfLiteSupportStatus::kInvalidArgumentError);
    }

    if (measure == tflite::scann_ondevice::core::UNSPECIFIED) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "UNSPECIFIED is not a valid value for ScaNN config DistanceMeasure.",
          TfLiteSupportStatus::kInvalidArgumentError);
    }

    // Make sure the query distance in different places are consistent.
    if (config.has_partitioner()) {
      DistanceMeasure partitioner_measure =
          config.partitioner().query_distance();
      if (measure != partitioner_measure) {
        return CreateStatusWithPayload(
            absl::StatusCode::kInvalidArgument,
            absl::StrFormat("DistanceMeasure %s is different from "
                            "DistanceMeasure %s found in partitioner config.",
                            DistanceMeasure_Name(measure),
                            DistanceMeasure_Name(partitioner_measure)),
            TfLiteSupportStatus::kInvalidArgumentError);
      }
    }
  }
  return measure;
}

absl::Status ConvertEmbeddingToEigenMatrix(const Embedding& embedding,
                                           Eigen::MatrixXf* matrix) {
  if (embedding.feature_vector().value_float().empty()) {
    // This should be caught upstream at EmbeddingPostprocessor creation.
    return CreateStatusWithPayload(absl::StatusCode::kInternal,
                                   "Float query embedding is empty.",
                                   TfLiteSupportStatus::kError);
  }
  Eigen::Map<const Eigen::VectorXf> query_ptr(
      embedding.feature_vector().value_float().data(),
      embedding.feature_vector().value_float().size());
  matrix->resize(embedding.feature_vector().value_float().size(), 1);
  matrix->col(0) = query_ptr;
  return absl::OkStatus();
}

}  // namespace

/* static */
StatusOr<std::unique_ptr<EmbeddingSearcher>> EmbeddingSearcher::Create(
    std::unique_ptr<SearchOptions> search_options,
    std::optional<absl::string_view> optional_index_file_content) {
  auto embedding_searcher = std::make_unique<EmbeddingSearcher>();


  RETURN_IF_ERROR(
      embedding_searcher->Init(
          std::move(search_options), optional_index_file_content));
  return embedding_searcher;
}




StatusOr<SearchResult> EmbeddingSearcher::Search(const Embedding& embedding) {
  // Convert embedding to Eigen matrix, as expected by ScaNN.
  Eigen::MatrixXf query;
  RETURN_IF_ERROR(ConvertEmbeddingToEigenMatrix(embedding, &query));


  // append to index
  //RETURN_IF_ERROR(AppendToIndex(embedding,"{'artist': 'Quinten', 'title': 'Quinten', 'url': 'https://qntn.nl'}"));

  // Identify partitions to search.
  std::vector<std::vector<int>> leaves_to_search(
      1, std::vector<int>(num_leaves_to_search_, -1));

  if (!partitioner_->Partition(query, &leaves_to_search)) {
    return CreateStatusWithPayload(absl::StatusCode::kInternal,
                                   "Partitioning failed.",
                                   TfLiteSupportStatus::kError);
  }

  // Prepare search results.
  std::vector<TopN> top_n;
  top_n.emplace_back(
      options_->max_results(),
      std::make_pair(std::numeric_limits<float>::max(), kNoNeighborId));

  // Perform search.
  if (quantizer_) {
    RETURN_IF_ERROR(
        QuantizedSearch(query, leaves_to_search[0], absl::MakeSpan(top_n)));
  } else { 
    RETURN_IF_ERROR(
        LinearSearch(query, leaves_to_search[0], absl::MakeSpan(top_n)));
  }

  // Build results.
  SearchResult search_result;
  for (const auto& [distance, id] : top_n[0].Take()) {

    if (id == kNoNeighborId) {
      break;
    }
    ASSIGN_OR_RETURN(auto metadata, index_->GetMetadataAtIndex(id));
    NearestNeighbor* nearest_neighbor = search_result.add_nearest_neighbors();
    nearest_neighbor->set_distance(distance);
    nearest_neighbor->set_metadata(std::string(metadata));
  }
  return search_result;
}

StatusOr<absl::string_view> EmbeddingSearcher::GetUserInfo() {
  return index_->GetUserInfo();
}

absl::Status EmbeddingSearcher::Init(
    std::unique_ptr<SearchOptions> options,
    std::optional<absl::string_view> optional_index_file_content) {
  RETURN_IF_ERROR(SanityCheckOptions(*options));
  options_ = std::move(options);

  // Initialize index.
  absl::string_view index_file_content;

  if (options_->has_index_file()) {
    ASSIGN_OR_RETURN(
        index_file_handler_,
        ExternalFileHandler::CreateFromExternalFile(&options_->index_file()));
    index_file_content = index_file_handler_->GetFileContent();
  } else {
    if (!optional_index_file_content) {
      absl::Status status = CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "Index File Content is expected when index_file option is not set.");
      LOG(ERROR) << "EmbeddingSearcher: " << status;
      return status;
    }
    index_file_content = *optional_index_file_content;
    
  }
  ASSIGN_OR_RETURN(index_,
                   Index::CreateFromIndexBuffer(index_file_content.data(),
                                                index_file_content.size()));
  ASSIGN_OR_RETURN(index_config_, index_->GetIndexConfig());
  RETURN_IF_ERROR(SanityCheckIndexConfig(index_config_));
  // Get distance measure once and for all.
  ASSIGN_OR_RETURN(distance_measure_,
                   GetDistanceMeasure(index_config_.scann_config()));

  // Initialize partitioner.
  if (index_config_.scann_config().has_partitioner()) {
    partitioner_ = tflite::scann_ondevice::core::Partitioner::Create(
        index_config_.scann_config().partitioner());
    num_leaves_to_search_ = std::min(
        static_cast<int>(ceilf(
            partitioner_->NumPartitions() *
            index_config_.scann_config().partitioner().search_fraction())),
        partitioner_->NumPartitions());

  } else {
    partitioner_ = absl::make_unique<tflite::scann_ondevice::core::NoOpPartitioner>();
    num_leaves_to_search_ = partitioner_->NumPartitions();
  }

  // Initialize product quantizer if needed.
  if (index_config_.scann_config().has_indexer()) {

    quantizer_ = tflite::scann_ondevice::core::AsymmetricHashQuerier::Create(
        index_config_.scann_config().indexer().asymmetric_hashing());
    
  }

  return absl::OkStatus();
}

absl::Status EmbeddingSearcher::AppendToIndex(const Embedding& embedding, const std::string metadataString) {
  // Replace input by string to be able to append to the index.
  // Convert String to embedding

  // Convert embedding to Eigen matrix, as expected by ScaNN.
  Eigen::MatrixXf query;
  RETURN_IF_ERROR(ConvertEmbeddingToEigenMatrix(embedding, &query));

  // Identify partitions to add therefore we use 1. (maybe make it more optimal by fetching multiple partitions and eventually recluster todo)
  std::vector<std::vector<int>> leaves_to_search(
      1, std::vector<int>(1, -1));
  if (!partitioner_->Partition(query, &leaves_to_search)) {
    return CreateStatusWithPayload(absl::StatusCode::kInternal,
                                   "Partitioning failed.",
                                   TfLiteSupportStatus::kError);
  }

  int leaf_id_to_add = leaves_to_search[0][0];

  // leaves_to_search contaisn now the top partition for this entry therefore we have to add to this partitoion the embedding.
  // in case of quantinization
  int global_offset = index_config_.global_partition_offsets(leaf_id_to_add);

  /* QueryInfo query_info;
    if (!quantizer_->Process(query, &query_info)) {
      return CreateStatusWithPayload(absl::StatusCode::kInternal,
                                    "Query quantization failed.",
                                    TfLiteSupportStatus::kError);
  }*/


  // get AsymmetricHashingProto from index_config
  tflite::scann_ondevice::core::AsymmetricHashingProto ah_proto = index_config_.scann_config().indexer().asymmetric_hashing();
  ah_proto.set_query_distance(tflite::scann_ondevice::core::DOT_PRODUCT);

  tflite::scann_ondevice::core::AsymmetricHashingIndexer indexer = tflite::scann_ondevice::core::AsymmetricHashingIndexer(ah_proto);
  std::vector<uint8_t> result(index_config_.embedding_dim(), 1);
  indexer.EncodeDatapoint(query, absl::MakeSpan(result));


  int total_partition_size = 0;
  int knumembeddings = 0;
  
  for (int i = 0; i < partitioner_->NumPartitions(); i++) {
    ASSIGN_OR_RETURN(auto partition, index_->GetPartitionAtIndex(i));
    int items_in_partition = (partition.size() / index_config_.embedding_dim());
    total_partition_size += partition.size();
    knumembeddings += items_in_partition;
  }

  knumembeddings += result.size() / index_config_.embedding_dim();
  total_partition_size += result.size();

  std::vector<uint8_t> database;
  database.reserve(knumembeddings * index_config_.embedding_dim());
  
  for (int i = 0; i < partitioner_->NumPartitions(); i++) {
    ASSIGN_OR_RETURN(auto partition, index_->GetPartitionAtIndex(i));

    // merge partition into database
    database.insert(database.end(), partition.begin(), partition.end());
  }
  // how to merge result into database
  database.insert(database.end(), result.begin(), result.end());

  std::vector<uint32_t> partition_assignment;
  partition_assignment.reserve(knumembeddings);

  std::vector<std::string> metadata;
  metadata.reserve(knumembeddings);

 for (int i = 0; i < partitioner_->NumPartitions(); i++) {
    ASSIGN_OR_RETURN(auto partition, index_->GetPartitionAtIndex(i));
    int items_in_partition = (partition.size() / index_config_.embedding_dim());

    // get offset
    int global_offset = index_config_.global_partition_offsets(i);

    // iterate 
    for(int j = 0; j < items_in_partition; j++) {
      partition_assignment.push_back(i);
      int index = j + global_offset;
      ASSIGN_OR_RETURN(absl::string_view m, index_->GetMetadataAtIndex(index));


      //push string format m
       metadata.push_back(std::string(m));
    }
  }

  partition_assignment.push_back(leaf_id_to_add);
  metadata.push_back(metadataString);

  // adjust config
  // remove all  index_config_.global_partition_offsets
  index_config_.clear_global_partition_offsets();
  
  
  // initialze IndexedArtifacts object
  tflite::scann_ondevice::IndexedArtifacts artifacts;
  artifacts.config = index_config_.scann_config();
  artifacts.embedding_dim = index_config_.embedding_dim();
  artifacts.hashed_database = database;
  artifacts.partition_assignment = partition_assignment;
  artifacts.metadata = metadata;
  artifacts.userinfo = "hashed_userinfo";

  ASSIGN_OR_RETURN(auto buffer,  tflite::scann_ondevice::CreateIndexBuffer(artifacts,true));

  // If we would like to overwrite The buffer..
  // clear index
  //index_.reset();  

  //ASSIGN_OR_RETURN(index_, Index::CreateFromIndexBuffer(buffer.data(), buffer.size()).value());


//   


if (options_->has_index_file()) {

  // overwrite file given by &options_->index_file() with buffer


    std::ofstream ofs("/data/user/0/nl.tudelft.trustchain/files/index_new_3.ldb", std::ofstream::out | std::ofstream::trunc);
    ofs << buffer;
    ofs.close();


    //std::unique_ptr<core::ExternalFile> external_file_ = std::make_unique<core::ExternalFile>();
    //external_file_->set_file_content(buffer);

    //ASSIGN_OR_RETURN(
    //index_file_handler_,
    //ExternalFileHandler::CreateFromExternalFile(external_file_.get()));
  

  //absl::Status status = index_file_handler_.OverwriteIndexFile(new_index_buffer);

    // check for okstatus and return
   // RETURN_IF_ERROR(index_file_handler_->WriteData(buffer));
    
    //index_file_content = index_file_handler_->GetFileContent();
  }
 

  //SetContents("/host_dir/index_new_q.ldb", buffer);
  //absl::string_view index_file_content;

   //index_file_content = index_file_handler_->GetFileContent();

  /*
  SetContents("index.ldb", buffer);

 
  tflite::task::core::ExternalFile file;
  file.set_file_name("index.ldb");

  // destory buffer
  //buffer.clear();

*/

 /* ASSIGN_OR_RETURN(
      index_file_handler_,
      ExternalFileHandler::CreateFromExternalFile(&options_->index_file()));
    index_file_content = index_file_handler_->GetFileContent();
*/
  // Initialize index.
  //absl::string_view index_file_content;

  //  ASSIGN_OR_RETURN(
  //  index_file_handler_,
  //  ExternalFileHandler::CreateFromExternalFile(&options_->index_file()));
  //  index_file_content = index_file_handler_->GetFileContent();
  // overwrite index_ with new initiliazed index from buffer
  index_ = Index::CreateFromIndexBuffer(buffer.data(), buffer.size()).value();
  ASSIGN_OR_RETURN(index_config_, index_->GetIndexConfig());

  RETURN_IF_ERROR(SanityCheckIndexConfig(index_config_));

    // Initialize partitioner.
  if (index_config_.scann_config().has_partitioner()) {
    partitioner_ = tflite::scann_ondevice::core::Partitioner::Create(
        index_config_.scann_config().partitioner());
    num_leaves_to_search_ = std::min(
        static_cast<int>(ceilf(
            partitioner_->NumPartitions() *
            index_config_.scann_config().partitioner().search_fraction())),
        partitioner_->NumPartitions());

  } else {
    partitioner_ = absl::make_unique<tflite::scann_ondevice::core::NoOpPartitioner>();
    num_leaves_to_search_ = partitioner_->NumPartitions();
  }

  // Initialize product quantizer if needed.
  if (index_config_.scann_config().has_indexer()) {

    quantizer_ = tflite::scann_ondevice::core::AsymmetricHashQuerier::Create(
        index_config_.scann_config().indexer().asymmetric_hashing());
    
  }

  return absl::OkStatus();

}

absl::Status EmbeddingSearcher::QuantizedSearch(
    Eigen::Ref<Eigen::MatrixXf> query, std::vector<int> leaves_to_search,
    absl::Span<TopN> top_n) {

  int dim = index_config_.embedding_dim();


  // Prepare QueryInfo used for all leaves.
  QueryInfo query_info;
  if (!quantizer_->Process(query, &query_info)) {
    return CreateStatusWithPayload(absl::StatusCode::kInternal,
                                   "Query quantization failed.",
                                   TfLiteSupportStatus::kError);
  }

  for (int leaf_id : leaves_to_search) {

    // Load partition into Eigen matrix.
    ASSIGN_OR_RETURN(auto partition, index_->GetPartitionAtIndex(leaf_id));

    int partition_size = partition.size() / dim;
    Eigen::Map<const Matrix8u> database(
      reinterpret_cast<const uint8_t*>(partition.data()), dim,
      partition_size);


    int global_offset = index_config_.global_partition_offsets(leaf_id);
    
    if (!AsymmetricHashFindNeighbors(query_info, database, global_offset,top_n)) {
      return CreateStatusWithPayload(absl::StatusCode::kInternal,
                     "Nearest neighbor search failed.",
                     TfLiteSupportStatus::kError);
    } 

  }
  return absl::OkStatus();
}

absl::Status EmbeddingSearcher::LinearSearch(Eigen::Ref<Eigen::MatrixXf> query,
                                             std::vector<int> leaves_to_search,
                                             absl::Span<TopN> top_n) {

  std::cout << "LinearSearch:\n";
  int dim = index_config_.embedding_dim();
  for (int leaf_id : leaves_to_search) {
    // Load partition into Eigen matrix.
    ASSIGN_OR_RETURN(auto partition, index_->GetPartitionAtIndex(leaf_id));
    int partition_size = partition.size() / (dim * sizeof(float));
    Eigen::Map<const Eigen::MatrixXf> database(
        reinterpret_cast<const float*>(partition.data()), dim, partition_size);
    // Perform search.
    int global_offset = index_config_.global_partition_offsets(leaf_id);
    if (!FloatFindNeighbors(query, database, global_offset, distance_measure_,
                            top_n)) {
      return CreateStatusWithPayload(absl::StatusCode::kInternal,
                                     "Nearest neighbor search failed.",
                                     TfLiteSupportStatus::kError);
    }
  }
  return absl::OkStatus();
}

}  // namespace processor
}  // namespace task
}  // namespace tflite
