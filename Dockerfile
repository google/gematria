FROM ubuntu:22.04
RUN apt-get update && apt-get install -y clang python3 python3-pip git curl
ARG bazelisk_version=1.17.0
RUN curl -L https://github.com/bazelbuild/bazelisk/releases/download/v${bazelisk_version}/bazelisk-linux-amd64 > /usr/bin/bazelisk && chmod +x /usr/bin/bazelisk && ln -s /usr/bin/bazelisk /usr/bin/bazel
WORKDIR /granlte
COPY . .
RUN pip3 install -r requirements.in