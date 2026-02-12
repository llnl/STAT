#!/bin/bash
set -x

# download and build launchmon
git clone https://github.com/llnl/launchmon
pushd launchmon
    ./bootstrap
    ./configure --prefix=$HOME/local --with-test-rm=slurm --with-test-rm-launcher=srun  || cat config.log
    make -j
    make install
popd

# download and build dyninst
wget -q https://github.com/dyninst/dyninst/archive/refs/tags/v11.0.0.tar.gz
tar xf v11.0.0.tar.gz
pushd dyninst-11.0.0
    mkdir build
    pushd build
        cmake -D CMAKE_INSTALL_PREFIX=$HOME/local ..
        cat CMakeFiles/CMakeOutput.log
        cat CMakeFiles/CMakeError.log
        make -j
        make install
    popd
popd

export CC="gcc"
export CXX="g++"
# download and build graphlib
wget -q https://github.com/LLNL/graphlib/archive/refs/tags/v3.0.0.tar.gz
tar xf v3.0.0.tar.gz
pushd graphlib-3.0.0
    mkdir build
    pushd build
        cmake -D CMAKE_INSTALL_PREFIX=$HOME/local ..
        make -j
        make install
    popd
popd

# download and build mrnet
#wget -q https://github.com/dyninst/mrnet/archive/refs/tags/v5.0.1.tar.gz
#tar xf v5.0.1.tar.gz
#pushd mrnet-5.0.1
git clone https://github.com/dyninst/mrnet
pushd mrnet
    ./configure --enable-shared --prefix=$HOME/local --disable-ltwt-threadsafe
    make
    make install
popd

