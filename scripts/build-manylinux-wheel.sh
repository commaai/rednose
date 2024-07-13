#!/bin/bash

set -ex

if [ -x /usr/bin/yum ]; then
    yum install -y python${PYTHON_VERSION}-devel.x86_64 clang libffi-devel eigen3-devel
fi

if [ -e /tmp/wheels ]; then
    echo "/tmp/wheels should not exist!"
    exit 1
fi

mkdir -p /tmp/wheels

${PYPATH}/bin/python -m pip wheel -v /project -w /tmp/wheels --no-deps
wheel=$(ls /tmp/wheels/*.whl)

# Apply fixups.
auditwheel repair ${wheel} -w /project/dist
