#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

cd $DIR

SUDO=""

# Use sudo if not root
if [[ ! $(id -u) -eq 0 ]]; then
  if [[ -z $(which sudo) ]]; then
    echo "Please install sudo or run as root"
    exit 1
  fi
  SUDO="sudo"
fi

$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends capnproto libcapnp-dev clang wget git autoconf libtool curl make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl libeigen3-dev python3-pip python3-dev

