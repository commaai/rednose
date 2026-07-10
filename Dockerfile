FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y clang wget git autoconf libtool curl make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl python3-pip python3-dev

WORKDIR /project

ENV PYTHONPATH=/project

COPY . .
RUN rm -rf .git
RUN pip3 install --break-system-packages --no-cache-dir '.[dev]'
RUN scons -c && scons -j$(nproc)
