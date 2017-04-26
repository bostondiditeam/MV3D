import model as mod
import glob
from config import *
import utils.batch_loading as ub


dataset_dir = cfg.PREPROCESSED_DATA_SETS_DIR


dates_to_drivers = {'1': ['15']}
training = ub.batch_loading(dataset_dir, dates_to_drivers)

dates_to_drivers = {'1': ['15']}
validation = ub.batch_loading(dataset_dir, dates_to_drivers)

m3=mod.MV3D()
m3.train(max_iter=10000, pre_trained=True,train_set=training,validation_set=validation)
