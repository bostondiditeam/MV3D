*This repo is for implementing MV3D from this paper: https://arxiv.org/abs/1611.07759 * 

*The MV3D implementation progress report can be found here*https://docs.google.com/document/d/1V-Go2kCxD58CIXKz3yk34pWeOnLca-0gqOw1JmfchrA/edit?usp=sharing

# Contents
- Key facts
- Workflow
- How to run
- Todos
- Issues

# Key facts

# Workflow

# Key Dependency
- A Nvidia GPU card with computation capability > 3
- CUDA
- Python3.5
- Tensorflow-GPU(version>1.0)


# How to run
*After Tensorflow-GPU could work*
If you are not using Nvidia K520 GPU, you need to change "arch=sm_30" in src/net/lib/setup.py and src/lib/make.sh in order to compiler *.so file right. 
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
cd src
source activate didi
sudo chmod 755 ./make.sh
./make.sh
python data.py # for process raw data to input network input format
python trainer.py # training the network. 
```

# Issue
- Not related to this repo, but if you are using Amazon CarND AWS AMI (Ubuntu 16.04 and with tensorflow-gpu 0.12 
installed),
 pip install --upgrade tensorflow **won't** work and will introduce driver/software conflict. Because CarND AMI has a
  nvidia 367 driver, but after running above line, it will install 375 driver. I think in this case, tensorflow-gpu
  (version >1.0)
  need to compiled from source code. 
- If you already have a Tensorflow-GPU > 1, then the above `./make.sh` works.
