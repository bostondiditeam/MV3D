#https://github.com/rbgirshick/fast-rcnn/blob/90e75082f087596f28173546cba615d41f0d38fe/lib/fast_rcnn/config.py

"""MV3D config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
import os.path as osp
import numpy as np
from time import strftime, localtime
from easydict import EasyDict as edict



__C = edict()
# Consumers can get config by:
#    import config as cfg
cfg = __C
__C.TEST_KEY=11

# dataset type
__C.DATA_SETS_TYPE='kitti'       #['didi','kitti','test']

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))

if __C.DATA_SETS_TYPE=='test':
    __C.DATA_SETS_DIR = osp.abspath('/home/stu/round12_data_test')
else:
    __C.DATA_SETS_DIR=osp.join(__C.ROOT_DIR,'data')

__C.RAW_DATA_SETS_DIR = osp.join(__C.DATA_SETS_DIR, 'raw', __C.DATA_SETS_TYPE)
__C.PREPROCESSED_DATA_SETS_DIR = osp.join(__C.DATA_SETS_DIR, 'preprocessed', __C.DATA_SETS_TYPE)
__C.PREPROCESSING_DATA_SETS_DIR = osp.join(__C.DATA_SETS_DIR, 'preprocessing', __C.DATA_SETS_TYPE)
__C.PREDICTED_XML_DIR = osp.join(__C.DATA_SETS_DIR, 'predicted', __C.DATA_SETS_TYPE)

__C.CHECKPOINT_DIR=osp.join(__C.ROOT_DIR,'checkpoint')
__C.LOG_DIR=osp.join(__C.ROOT_DIR,'log')

__C.USE_RESNET_AS_TOP_BASENET = True

__C.IMAGE_FUSION_DIABLE = False
__C.RGB_BASENET = 'resnet'  # 'resnet' 、'xception' 'VGG'
if __C.RGB_BASENET == 'xception':
    __C.USE_IMAGENET_PRE_TRAINED_MODEL = True
else:
    __C.USE_IMAGENET_PRE_TRAINED_MODEL =False

__C.TRACKLET_GTBOX_LENGTH_SCALE = 1.6

# image crop config
if __C.DATA_SETS_TYPE ==  'didi' or __C.DATA_SETS_TYPE   ==  'test':
    __C.IMAGE_CROP_LEFT     =0 #pixel
    __C.IMAGE_CROP_RIGHT    =0
    __C.IMAGE_CROP_TOP      =400
    __C.IMAGE_CROP_BOTTOM   =100
else:
    __C.IMAGE_CROP_LEFT     =0  #pixel
    __C.IMAGE_CROP_RIGHT    =0
    __C.IMAGE_CROP_TOP      =0
    __C.IMAGE_CROP_BOTTOM   =0

# image
if __C.DATA_SETS_TYPE   ==  'test':
    __C.IMAGE_HEIGHT=1096 #pixel
    __C.IMAGE_WIDTH=1368
elif __C.DATA_SETS_TYPE ==  'didi':
    __C.IMAGE_HEIGHT=1096 #pixel
    __C.IMAGE_WIDTH=1368
elif __C.DATA_SETS_TYPE == 'kitti':
    __C.IMAGE_WIDTH=1242
    __C.IMAGE_HEIGHT=375
# if timer is needed.
__C.TRAINING_TIMER = True
__C.TRACKING_TIMER = True
__C.DATAPY_TIMER = False

# print(cfg.RAW_DATA_SETS_DIR)
# print(cfg.PREPROCESSED_DATA_SETS_DIR)
# print(cfg.PREDICTED_XML_DIR)

__C.USE_CLIDAR_TO_TOP = False

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value

if __name__ == '__main__':
    print('__C.ROOT_DIR = '+__C.ROOT_DIR)
    print('__C.DATA_SETS_DIR = '+__C.DATA_SETS_DIR)
    print('__C.RAW_DATA_SETS_DIR = '+__C.RAW_DATA_SETS_DIR)