git submodule update --init --recursive
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
cd geant4/cpp
if [ -d "build" ]; then rm -rf build; fi
mkdir build
cd build
cmake -Dpybind11_DIR=/usr/local/lib/python3.10/dist-packages/pybind11/share/cmake/pybind11 -DPython_EXECUTABLE=/usr/bin/python3 ..
make -j
cd ../../..


