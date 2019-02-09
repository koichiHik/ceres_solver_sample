#!/bin/bash

export CC=/usr/bin/clang CXX=/usr/bin/clang++

if [ ! -e ./build ]; then
  mkdir build
fi

cd build

cmake ../

make

cd ../
