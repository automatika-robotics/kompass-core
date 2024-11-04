# exit immediately on any failed step
set -xe

# install vcpkg
git clone --depth 1 https://github.com/microsoft/vcpkg.git
yum install -y zip unzip
cd vcpkg && ./bootstrap-vcpkg.sh -disableMetrics
export VCPKG_ROOT=$PWD

# install dependencies
./vcpkg install fcl
./vcpkg install pcl
./vcpkg install ompl
cd ..

# install python-devel (for PyBind11)
yum install -y python39-devel
