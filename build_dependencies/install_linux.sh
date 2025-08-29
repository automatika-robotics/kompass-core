# exit immediately on any failed step
set -xe
mkdir -p $VCPKG_ROOT
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH

# install vcpkg
git clone --depth 1 https://github.com/microsoft/vcpkg.git $VCPKG_ROOT
yum install -y zip unzip ninja-build

if [ "$(uname -m)" != "x86_64" ]
then
    export VCPKG_FORCE_SYSTEM_BINARIES=1
fi

$VCPKG_ROOT/bootstrap-vcpkg.sh -disableMetrics

# install dependencies
$VCPKG_ROOT/vcpkg install fcl --triplet=$VCPKG_TARGET_TRIPLET
$VCPKG_ROOT/vcpkg install ompl --triplet=$VCPKG_TARGET_TRIPLET
