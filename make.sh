sudo chmod 777 ./ -R
cd ./src/
cd ./net/lib/
python ./setup.py build_ext --inplace
./make.sh
cd ../../
cd lidar_data_preprocess/Python_to_C_Interface/ver3
make
cd ../../../ #back to src