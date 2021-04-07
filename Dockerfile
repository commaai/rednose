from ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y libzmq3-dev capnproto libcapnp-dev clang wget git autoconf libtool curl make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl libeigen3-dev

RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
ENV PATH="/root/.pyenv/bin:/root/.pyenv/shims:${PATH}"
RUN pyenv install 3.8.5
RUN pyenv global 3.8.5
RUN pyenv rehash
RUN pip3 install scons==4.1.0.post1 pre-commit==2.10.1 pylint==2.7.1 Cython==0.29.22

WORKDIR /project

ENV PYTHONPATH=/project

COPY . .
RUN rm -rf .git
RUN python3 setup.py install
RUN scons -c && scons -j$(nproc)
