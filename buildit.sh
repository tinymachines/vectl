#!/bin/bash

rm -rf build/ dist/ *.egg-info/
python setup.py build_ext --inplace -v
#mkdir ./build
#make clean
#cd build
#cmake ..
#make
#sudo make install

