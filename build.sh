#!/bin/bash
# Build the project into python and cpp libraries

mkdir -p build
rm -rf build/*
cd build
cmake ..
make -j4
cd ..

echo "Finished Build"