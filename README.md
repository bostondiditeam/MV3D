* This repo is for implementing MV3D from this paper: https://arxiv.org/abs/1611.07759 * 

* The MV3D implementation progress report can be found [here](https://docs.google.com/document/d/1V-Go2kCxD58CIXKz3yk34pWeOnLca-0gqOw1JmfchrA/edit?usp=sharing) 


To clone,

    $git clone --recursive  https://github.com/bostondiditeam/MV3D.git

To pull, use 

    $git pull --recurse-submodules  
    

# Contents
- Key facts
- Workflow
- How to run
- Todo
- Issues

# Key facts

# Workflow
Please refer to [here](https://drive.google.com/file/d/0B47OwD8fbxJNWWFmVUJ4OUV4RUE/view?usp=sharing)

# Key Dependency
- A Nvidia GPU card with computation capability > 3
- CUDA
- Python3.5 for MV3D related code
- Tensorflow-GPU(version>1.0)
- Python2.7 for ROS related script

# File Structure
```
├── data   <-- all data is stored here. (Introduced in detail below)
│   ├── predicted  <-- after prediction, results will be saved here.
│   ├── preprocessed   <-- MV3D net will take inputs from here(after data.py) 
│   └── raw <-- raw data
├── environment_cpu.yml  <-- install cpu version.
├── README.md
├── saved_model                 <--- model and weights saved here. 
├── src        <-- MV3D net related source code 
│   ├── config.py
│   ├── data.py
│   ├── didi_data
│   ├── kitti_data
│   ├── lidar_data_preprocess
│   ├── make.sh
│   ├── model.py
│   ├── mv3d_net.py
│   ├── net
│   ├── play_demo.ipynb
│   ├── __pycache__
│   ├── tracking.py   <--- prediction after training. 
│   ├── tracklets
│   └── train.py    <--- training the whole network. 
│── utils    <-- all related tools put here, like ros bag data into kitti format
│    └── bag_to_kitti  <--- Take lidar value from ROS bag and save it as bin files.
└── external_models    <-- use as a submodule, basically code from other repos.
    └── didi-competition  <--- Code from Udacity's challenge repo with slightly modification, sync with Udacity's new
     updates regularly. 
```

# Related data are organized in this way. (Under /data directory)
```
├── predicted <-- after prediction, results will be saved here.
│   ├── didi <-- when didi dataset is used, the results will be put here
│   └── kitti <-- When kitti dataset used for prediction, put the results here
│       ├── iou_per_obj.csv   <-- What will be evaluated for this competition, IoU score
│       ├── pr_per_iou.csv   <--precision and recall rate per iou, currently not be evaluated by didi's rule
│       └── tracklet_labels_pred.xml  <-- Tracklet generated from prediction pipeline. 
├── preprocessed  <-- Data will be fed into MV3D net (After processed by data.py)
│   ├── didi <-- When didi dataset is processed, save it here
│   └── kitti <-- When Kitti dataset is processed, save it here
│       ├── gt_boxes3d
│           └── 2011_09_26
│               └── 0005
|                   |___ 00000.npy
├       |── gt_labels
│           └── 2011_09_26
│               └── 0005 
|                   |___ 00000.npy
|       ├── rgb
│           └── 2011_09_26
│               └── 0005 
|                   |___ 00000.png
|       ├── top
│           └── 2011_09_26
│               └── 0005 
|                   |___ 00000.npy
|       └── top_image
|           └── 2011_09_26
|               └── 0005 
|                   |___ 00000.png
└── raw  <-- this strictly follow KITTI raw data file format, while seperated into didi and kitti dataset. 
    ├── didi <-- will be something similar to kitti raw data format below. 
    └── kitti
        └── 2011_09_26
            ├── 2011_09_26_drive_0005_sync
            │   ├── image_02
            │   │   ├── data
            │   │   │   └── 0000000000.png
            │   │   └── timestamps.txt
            │   ├── tracklet_labels.xml
            │   └── velodyne_points
            │       ├── data
            │       │   └── 0000000000.bin
            │       ├── timestamps_end.txt
            │       ├── timestamps_start.txt
            │       └── timestamps.txt
            ├── calib_cam_to_cam.txt
            ├── calib_imu_to_velo.txt
            └── calib_velo_to_cam.txt

```

# Modification needed to run
*After Tensorflow-GPU could work*
If you are not using Nvidia K520 GPU, you need to change "arch=sm_30" to other value in src/net/lib/setup.py and src/lib/make.sh in order to compiler *.so file right. 
Here is  short list for arch values for different architecture. 

```
# Which CUDA capabilities do we want to pre-build for?
# https://developer.nvidia.com/cuda-gpus
#   Compute/shader model   Cards
#   6.1		      P4, P40, Titan X so CUDA_MODEL = 61
#   6.0                    P100 so CUDA_MODEL = 60
#   5.2                    M40
#   3.7                    K80
#   3.5                    K40, K20
#   3.0                    K10, Grid K520 (AWS G2)
#   Other Nvidia shader models should work, but they will require extra startup
#   time as the code is pre-optimized for them.
CUDA_MODELS=30 35 37 52 60 61
```
Test your Tensorflow-GPU is running by"
```
import tensorflow as tf
sess = tf.Session()
print(tf.__version__) # version more than v1. 
```
It runs without error message and show　＂successfully opened CUDA library libcublas.so.8.0 locally＂, then it is in CUDA successfully.


```
source activate didi
sudo chmod 755 ./make.sh
./make.sh
# prerequisite for next step, i.e. running preprocessing using data.py, is to 
# follow steps in utils/bag_to_kitti if using didi data
python data.py # for process raw data to input network input format
python train.py # training the network. 
```

# Some other readme.md files inside this repo:
- How to extract and sync data from ROS bags [Under utils/bag_to_kitti](./utils/bag_to_kitti/README.md)
- How to generate tracklet files [Under src/tracklets/](./src/tracklets/README.md) 


# Issue
- Not related to this repo, but if you are using Amazon CarND AWS AMI (Ubuntu 16.04 and with tensorflow-gpu 0.12 
installed),
 pip install --upgrade tensorflow **won't** work and will introduce driver/software conflict. Because CarND AMI has a
  nvidia 367 driver, but after running above line, it will install 375 driver. I think in this case, tensorflow-gpu
  (version >1.0)
  need to compiled from source code. 
- If you already have a Tensorflow-GPU > 1, then the above `./make.sh` works.
- If you see error message "tensorflow.python.framework.errors_impl.NotFoundError: YOUR_FOLDER/roi_pooling.so: undefined symbol: _ZN10tensorflow7strings6StrCatB5cxx11ERKNS0_8AlphaNumES3_", it is related to compilation of roi_pooling layer. A simple fix will be changing "GLIBCXX_USE_CXX11_ABI=1" to "GLIBCXX_USE_CXX11_ABI=0" in "src/net/lib/make.sh" (line 17)
