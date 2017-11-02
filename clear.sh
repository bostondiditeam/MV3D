cd ./src/

rm ./net/lib/psroi_pooling_layer/psroi_pooling.so
rm ./net/lib/pycocotools/_mask.cpython-35m-x86_64-linux-gnu.so


#.c
rm ./net/lib/nms/cpu_nms.c
rm ./net/lib/utils/bbox.c

#.cpp
rm ./net/lib/nms/gpu_nms.cpp

rm -Rf ./net/lib/build
