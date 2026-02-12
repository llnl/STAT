#!/bin/bash
#
#

set -ex

prefix=$HOME/local
cachedir=$HOME/local/.cache
components=all

#
# Ordered list of software to download and install into $prefix.
#  NOTE: Code currently assumes .tar.gz suffix...
#
#https://github.com/LLNL/LaunchMON/releases/download/v1.0.2/launchmon-v1.0.2.tar.gz \
# NOTE: openmpi and dyninst both take a long time to build and will likely cause travis to timeout after 50 minutes, so they should be built in two passes
downloads="\
https://www.gnupg.org/ftp/gcrypt/libgpg-error/libgpg-error-1.36.tar.bz2 \
https://www.gnupg.org/ftp/gcrypt/libgcrypt/libgcrypt-1.8.5.tar.bz2 \
https://github.com/LLNL/graphlib/archive/v3.0.0.tar.gz \
http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/16120/l_mpi_2019.6.166.tgz \
https://github.com/dyninst/dyninst/archive/v10.2.1.tar.gz \
https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.2.tar.gz \
"
#https://github.com/LLNL/LaunchMON/releases/download/v1.0.2/launchmon-v1.0.2.tar.gz\
#https://github.com/dyninst/mrnet/archive/v5.0.1.tar.gz \

checkouts="\
https://github.com/dyninst/mrnet.git \
https://github.com/llnl/launchmon.git
"

declare -A checkout_sha1=(\
["launchmon"]="63e254697808c9376a3638ed2166b795104d51aa"
)

declare -A extra_configure_opts=(\
["launchmon-v1.0.2"]="--with-test-rm=mpiexec_hydra --with-test-ncore-per-CN=2 --with-test-nnodes=1 --with-test-rm-launcher=${prefix}/bin/mpirun --with-test-installed" \
["launchmon"]="--with-test-rm=mpiexec_hydra --with-test-ncore-per-CN=2 --with-test-nnodes=1 --with-test-rm-launcher=${prefix}/bin/mpirun --with-test-installed" \
["v5.0.1"]="--enable-shared" \
["mrnet"]="--enable-verbosebuild --disable-ltwt-threadsafe --enable-shared" \
["openmpi-4.0.2"]="--enable-orterun-prefix-by-default" \
)

# we install dyninst in a separate prefix b/c otherwise it will break on previously installed headers
declare -A extra_cmake_opts=(\
["v10.2.1"]="-DCMAKE_BUILD_TYPE=Debug"
)

declare -r prog=${0##*/}
declare -r long_opts="prefix:,cachedir:,verbose,printenv,components:"
declare -r short_opts="vp:c:PC:"
declare -r usage="\
\n
Usage: $prog [OPTIONS]\n\
Download and install to a local prefix (default=$prefix) dependencies\n\
for building STAT\n\
\n\
Options:\n\
 -v, --verbose           Be verbose.\n\
 -P, --printenv          Print environment variables to stdout\n\
 -c, --cachedir=DIR      Check for precompiled dependency cache in DIR\n\
 -e, --max-cache-age=N   Expire cache in N days from creation\n\
 -p, --prefix=DIR        Install software into prefix\n
 -C, --components=VAL    VAL=[ompi|dyninst|other]\n
"

die() { echo -e "$prog: $@"; exit 1; }
say() { echo -e "$prog: $@"; }
debug() { test "$verbose" = "t" && say "$@"; }

GETOPTS=`/usr/bin/getopt -u -o $short_opts -l $long_opts -n $prog -- $@`
if test $? != 0; then
    die "$usage"
fi

eval set -- "$GETOPTS"
while true; do
    case "$1" in
      -v|--verbose)          verbose=t;       shift   ;;
      -c|--cachedir)         cachedir="$2";   shift 2 ;;
      -e|--max-cache-age)    cacheage="$2";   shift 2 ;;
      -p|--prefix)           prefix="$2";     shift 2 ;;
      -P|--printenv)         print_env=1;     shift   ;;
      -C|--components)       components="$2"; shift 2 ;;
      --)                    shift ; break;         ;;
      *)                     die "Invalid option '$1'\n$usage" ;;
    esac
done

print_env () {
    echo "export LD_LIBRARY_PATH=${prefix}/lib:$LD_LIBRARY_PATH"
    echo "export CPPFLAGS=-I${prefix}/include"
    echo "export LDFLAGS=-L${prefix}/lib"
    echo "export PATH=${PATH}:${prefix}/usr/bin:${prefix}/bin"
}

if test -n "$print_env"; then
    print_env
    exit 0
fi

eval $(print_env)

sanitize ()
{
    # Sanitize cache name
    echo $1 | sed 's/[\t /\]/_/g'
}

check_cache ()
{
    test -n "$cachedir" || return 1
    local url=$(sanitize $1)
    local cachefile="${cachedir}/${url}"
    test -f "${cachefile}" || return 1
    test -n "$cacheage"    || return 0

    local ctime=$(stat -c%Y ${cachefile})
    local maxage=$((${cacheage}*86400))
    test $ctime -gt $maxage && return 1
}

add_cache ()
{
    test -n "$cachedir" || return 0
    mkdir -p "${cachedir}" &&
    touch "${cachedir}/$(sanitize ${1})"
}

env

for pkg in $downloads; do
    name=$(basename ${pkg} .tar.gz)
    name=$(basename ${name} .tar.bz2)
    name=$(basename ${name} .tgz)
    if test "$components" = "dyninst" -a "$name" != "v10.2.1"; then
      continue
    fi
    if test "$components" = "impi" -a "$name" != "l_mpi_2019.6.166"; then
      continue
    fi
    if test "$components" = "ompi" -a "$name" != "openmpi-4.0.2"; then
      continue
    fi
    if test "$components" = "other" -a "$name" = "l_mpi_2019.6.166"; then
      continue
    fi
    if test "$components" = "other" -a "$name" = "openmpi-4.0.2"; then
      continue
    fi
    if test "$components" = "other" -a "$name" = "v10.2.1"; then
      continue
    fi
    cmake_opts="${extra_cmake_opts[$name]}"
    configure_opts="${extra_configure_opts[$name]}"
    cache_name="$name:$sha1:$make_opts:$configure_opts:$cmake_opts"
    if check_cache "$name"; then
       say "Using cached version of ${name}"
       continue
    fi
    export CC=gcc-8
    export CXX=g++-8
    export ACLOCAL_PATH=${prefix}/share/aclocal
    export PATH=${prefix}/bin:$PATH
    export VERBOSE=1
    if test "$name" = "l_mpi_2019.6.166"; then
      curl -L -O --insecure ${pkg} || die "Failed to download ${pkg}"
      tar xf l_mpi_2019.6.166.tgz
      cd l_mpi_2019.6.166
      cat << EOF > install.ini
ACCEPT_EULA=accept
CONTINUE_WITH_OPTIONAL_ERROR=yes
PSET_MODE=install
PSET_INSTALL_DIR=${prefix}
CONTINUE_WITH_INSTALLDIR_OVERWRITE=yes
ARCH_SELECTED=ALL
COMPONENTS=ALL
EOF
      ./install.sh --silent install.ini
      ls -l ${prefix}/impi/2019.6.166/intel64/bin/
      for myfile in mpicc mpiCC mpicxx mpiexec mpiexec.hydra mpifc mpirun mpif77 mpif90 mpifort mpigcc mpigxx mpivars.sh mpivars.csh hydra_bstrap_proxy hydra_nameserver hydra_pmi_proxy impi_info cpuinfo; do
        rm -f ${prefix}/bin/${myfile}
        ln -s ${prefix}/impi/2019.6.166/intel64/bin/${myfile} ${prefix}/bin
      done
      add_cache "$name"
      continue
    fi
    mkdir -p ${name}  || die "Failed to mkdir ${name}"
    (
      cd ${name} &&
      curl -L -O --insecure ${pkg} || die "Failed to download ${pkg}"
      tar --strip-components=1 -xf *.tar.gz || tar --strip-components=1 -xf *.tar.bz2 || tar --strip-components=1 -xf *.tgz || die "Failed to un-tar ${name}"
      if test -x configure; then
        ./configure --prefix=${prefix} \
                    $configure_opts  || head config.log
      elif test -f CMakeLists.txt; then
        mkdir build && cd build
        which cmake
        cmake -DCMAKE_INSTALL_PREFIX=${prefix} $cmake_opts ..
      fi

      make -j 16 PREFIX=${prefix} || make PREFIX=${prefix}
      make -j 16 PREFIX=${prefix} install || make PREFIX=${prefix} install
      if test "$name" = "launchmon-v1.0.2"; then
        pushd test/src
        echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
        export PATH=./:${prefix}/bin:$PATH
        which mpirun
        export LMON_FE_ENGINE_TIMEOUT=60
        cat test.launch_1
        ./test.launch_1
        sleep 60
        cat test.attach_1
        ./test.attach_1
        sleep 60
        echo 'LMON test done'
        popd
      fi
    ) || die "Failed to build and install $name"
    add_cache "$name"
done


for url in $checkouts; do
    if test "$components" != "all" -a "$components" != "other"; then
      continue
    fi
    source $HOME/local/bin/mpivars.sh
    name=$(basename ${url} .git)
    sha1="${checkout_sha1[$name]}"
    make_opts="${extra_make_opts[$name]}"
    cmake_opts="${extra_cmake_opts[$name]}"
    configure_opts="${extra_configure_opts[$name]}"
    cache_name="$name:$sha1:$make_opts:$configure_opts:$cmake_opts"
    if check_cache "$cache_name"; then
       say "Using cached version of ${name}"
       continue
    fi
    git clone ${url} ${name} || die "Failed to clone ${url}"
    (
      cd ${name} || die "cd failed"
      if test -n "$sha1"; then
        git checkout $sha1
      fi

      if ! test -f configure; then
        export ACLOCAL_PATH=${prefix}/share/aclocal
        ./bootstrap
      fi
      if test -x configure; then
        ./configure --prefix=${prefix} \
                    $configure_opts || cat config.log
                    #--sysconfdir=${prefix}/etc \
      elif test -f CMakeLists.txt; then
          mkdir build && cd build
          cmake -DCMAKE_INSTALL_PREFIX=${prefix} $cmake_opts ..
      fi
# The hack below is only needed in the dist:trusty travis env
# This will work around a ptrace Input/Output error when removing breakpoints
      if test "$name" = "launchmon"; then
        sed -i launchmon/src/linux/sdbg_linux_launchmon.cxx -e '2975 s|disable|//disable|'
      fi
      make -j 16 PREFIX=${prefix} $make_opts || make PREFIX=${prefix} $make_opts
      make -j 16 PREFIX=${prefix} $make_opts install || make PREFIX=${prefix} $make_opts install
      #make -j 16 check PREFIX=${prefix} $make_opts || make check PREFIX=${prefix} $make_opts
#      if test "$name" = "launchmon"; then
#        pushd test/src
#        echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
#        export PATH=./:${prefix}/bin:$PATH
#        export LMON_FE_ENGINE_TIMEOUT=60
#        cat test.launch_1
#        ./test.launch_1
#        sleep 60
#        cat test.attach_1
#        ./test.attach_1
#        sleep 60
#        echo 'LMON test done'
#        popd
#      fi
    ) || die "Failed to build and install $name"
    ls -l ${prefix}/etc/ || echo failed
    cat ${prefix}/etc/rm_mpiexec_hydra.conf || echo "not found"
    add_cache "$cache_name"
done

