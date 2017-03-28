cd ./net/lib/
python ./setup.py build_ext --inplace
./make.sh
cd ../../

mv ./net/lib/roi_pooling_layer/roi_pooling.so ./net/roipooling_op/
mv ./net/lib/nms/gpu_nms.cpython-35m-x86_64-linux-gnu.so ./net/processing/
mv ./net/lib/utils/cython_bbox.cpython-35m-x86_64-linux-gnu.so ./net/processing

