#/bin/sh

if [[ "$PWD" =~ build ]]; then
    cd ..
fi
rm -r build
cmake -B build -DMLP_XMOS=ON -DCMAKE_TOOLCHAIN_FILE=xmos_cmake_toolchain/xs3a.cmake
pushd build
make MLP_XMOS
xrun --xscope --xscope-file xscope.vcd MLP_XMOS.xe
popd