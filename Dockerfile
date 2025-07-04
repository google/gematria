FROM ubuntu:24.04
RUN apt-get update && \
apt-get install -y \
    curl \
    git \
    libelf-dev \
    m4 \
    python3 \
    python3-pip \
    gnupg
RUN echo "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-19 main\ndeb-src http://apt.llvm.org/noble/ llvm-toolchain-noble-19 main" >> /etc/apt/sources.list && \
  curl -L https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
  curl -L https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc && \
  apt-get update && \
  apt-get install -y clang-19 lld-19 && \
  ln -s /usr/bin/clang-19 /usr/bin/clang && \
  ln -s /usr/bin/clang-19 /usr/bin/clang++
ARG bazelisk_version=1.19.0
RUN curl -L https://github.com/bazelbuild/bazelisk/releases/download/v${bazelisk_version}/bazelisk-linux-amd64 > /usr/bin/bazelisk && chmod +x /usr/bin/bazelisk && ln -s /usr/bin/bazelisk /usr/bin/bazel
WORKDIR /gematria
COPY . .
ENV USE_BAZEL_VERSION 7.x

