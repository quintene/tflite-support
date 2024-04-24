
# Download and move android SDK tools to specific folders
FROM tensorflow/build:latest-python3.11

ENV ANDROID_SDK_API_LEVEL 29
ENV ANDROID_API_LEVEL 29
ENV ANDROID_BUILD_TOOLS_VERSION 30.0.3
ENV ANDROID_DEV_HOME /android
ENV ANDROID_NDK_API_LEVEL 21
ENV ANDROID_NDK_FILENAME android-ndk-r19c-linux-x86_64.zip
ENV ANDROID_NDK_HOME /android/ndk
ENV ANDROID_NDK_URL https://dl.google.com/android/repository/android-ndk-r19c-linux-x86_64.zip
ENV ANDROID_SDK_FILENAME tools_r25.2.5-linux.zip
ENV ANDROID_SDK_HOME /android/sdk
ENV ANDROID_HOME /android/sdk
ENV ANDROID_SDK_URL https://dl.google.com/android/repository/tools_r25.2.5-linux.zip
RUN apt-get update 

# Create folders
RUN mkdir -p '/android/sdk'

RUN wget -q 'https://dl.google.com/android/repository/tools_r25.2.5-linux.zip'

RUN unzip 'tools_r25.2.5-linux.zip'
RUN mv 'tools' '/android/sdk'
# Copy paste the folder
RUN cp -r /android/sdk/tools /android/android-sdk-linux

# Download NDK, unzip and move contents
RUN wget 'https://dl.google.com/android/repository/android-ndk-r19c-linux-x86_64.zip'

RUN unzip 'android-ndk-r19c-linux-x86_64.zip'
RUN mv android-ndk-r19c ndk
RUN mv 'ndk' '/android'
# Copy paste the folder
RUN cp -r /android/ndk /android/android-ndk-r19c

# Remove .zip files
RUN rm 'tools_r25.2.5-linux.zip'
RUN rm 'android-ndk-r19c-linux-x86_64.zip'

# Make android ndk executable to all users
#RUN chmod -R go=u '/android'

RUN export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/tools/node/bin:/tools/google-cloud-sdk/bin:/opt/bin:/android/sdk/tools:/android/sdk/platform-tools:/android/ndk



