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

Test your Tensorflow-GPU is running by"
```
import tensorflow as tf
sess = tf.Session()
```
It runs without error message and show it opens cuda libraries. 
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
