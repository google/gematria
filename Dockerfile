FROM ubuntu:22.04
RUN apt-get update && \
apt-get install -y \
    clang \
    curl \
    git \
    libelf-dev \
    lld \
    m4 \
    python3 \
    python3-pip
ARG bazelisk_version=1.19.0
RUN curl -L https://github.com/bazelbuild/bazelisk/releases/download/v${bazelisk_version}/bazelisk-linux-amd64 > /usr/bin/bazelisk && chmod +x /usr/bin/bazelisk && ln -s /usr/bin/bazelisk /usr/bin/bazel
WORKDIR /gematria
COPY . .
ENV USE_BAZEL_VERSION 6.4.0
RUN pip3 install -r requirements.txt

