#/bin/sh

if [[ "$PWD" =~ build ]]; then
    cd ..
fi
rm -r build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
pushd build
make FuncLearnTest
#./FuncLearnTest_d
popd
