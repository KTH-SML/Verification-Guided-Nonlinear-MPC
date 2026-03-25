#!/usr/bin/env bash

set -e

echo "===== Cloning acados ====="
git clone https://github.com/acados/acados.git
cd acados

echo "===== Updating submodules ====="
git submodule update --recursive --init

echo "===== Creating build directory ====="
mkdir build
cd build

echo "===== Running CMake ====="
cmake -DACADOS_WITH_QPOASES=ON \
      ..

echo "===== Building ====="
make -j4

echo "===== Installing ====="
make install

echo "===== Installing Python interface ====="
cd ..
pip install -e interfaces/acados_template

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"./acados/lib"
export ACADOS_SOURCE_DIR="./acados"

echo "===== Done! ====="