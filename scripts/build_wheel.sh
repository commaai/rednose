#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

cd $DIR/..

export CIBW_BUILD="cp312-manylinux_x86_64 cp311-manylinux_x86_64"
export CIBW_MANYLINUX_X86_64_IMAGE="quay.io/pypa/manylinux_2_28_x86_64"
export CIBW_BEFORE_ALL_LINUX="dnf install -y clang libffi-devel eigen3-devel"

cibuildwheel
