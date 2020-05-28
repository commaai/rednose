from ubuntu:16.04

RUN apt-get update && apt-get install -y libzmq3-dev capnproto libcapnp-dev clang wget git autoconf libtool curl make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl libeigen3-dev

RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
ENV PATH="/root/.pyenv/bin:/root/.pyenv/shims:${PATH}"
RUN pyenv install 3.7.3
RUN pyenv global 3.7.3
RUN pyenv rehash
RUN pip3 install scons==3.1.1 pre-commit==2.4.0 pylint==2.5.2

WORKDIR /project/rednose

ENV PYTHONPATH=/project

COPY . .
RUN rm -rf .git
RUN python3 setup.py install
RUN scons -c && scons -j$(nproc)
