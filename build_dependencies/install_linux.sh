# exit immediately on any failed step
set -xe
mkdir -p $VCPKG_ROOT
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH

# install vcpkg
git clone --depth 1 https://github.com/microsoft/vcpkg.git $VCPKG_ROOT
yum install -y zip unzip
$VCPKG_ROOT/bootstrap-vcpkg.sh -disableMetrics

# install dependencies
$VCPKG_ROOT/vcpkg install fcl
$VCPKG_ROOT/vcpkg install pcl[core]
$VCPKG_ROOT/vcpkg install ompl

