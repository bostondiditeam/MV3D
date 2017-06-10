/* velo_ext.cpp */
#include "lidar_preprocess.hpp"

BOOST_PYTHON_MODULE(lidar_preprocess_ext)
{
    using namespace boost::python;
    def("lidar_preprocess", lidar_preprocess);
}
