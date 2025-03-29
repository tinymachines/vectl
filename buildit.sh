#!/bin/bash

rm -R ./build &>/dev/null
mkdir ./build
make clean
cd build
cmake ..
make
sudo make install

