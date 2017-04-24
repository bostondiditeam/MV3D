import model as mod
import glob
from config import *


dataset_dir = cfg.PREPROCESSED_DATA_SETS_DIR

# dates  = ['2011_09_26']
# drivers = ['0001', '0017', '0029', '0052', '0070', '0002', '0018', '0035', '0056', '0079', '0005', '0019', '0036',
#            '0057', '0084', '0009', '0020', '0039', '0059', '0086', '0011', '0023', '0046', '0060', '0091', '0013',
#            '0027', '0048', '0061', '0015', '0028', '0051', '0064']
# load_indexs = None

dates  = ['1']
drivers = ['15']
load_indexs = None

# if load single images for testing.
# load_indexs = ['2011_09_26_0005_00000', '2011_09_26_0005_00010','2011_09_26_0005_00020', '2011_09_26_0005_00030',
#                '2011_09_26_0005_00040', '2011_09_26_0005_00050']

#
m3=mod.MV3D()
m3.train(max_iter=10000, pre_trained=True, dataset_dir=dataset_dir, dates=dates, drivers=drivers,
         frames_index =load_indexs)
