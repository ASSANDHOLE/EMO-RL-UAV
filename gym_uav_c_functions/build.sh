#!/bin/bash

set -e

if [ ! -d "build" ]
then
  mkdir build
else
  rm -rf ./build
  mkdir build
fi

cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j 8
cd ..
cp ./build/libc_functions.so ../gym_uav/env/
echo "copy 'libc_functions.so' to ../gym_uav/env/"