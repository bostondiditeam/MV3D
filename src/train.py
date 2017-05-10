import model as mod
import glob
from config import *
import utils.batch_loading as ub


dataset_dir = cfg.PREPROCESSING_DATA_SETS_DIR


training_dataset = {
    '1': ['6_f', '9_f', '15', '20']
#    '2': ['3_f'],
#    '3': ['2_f','4','6','8','7']}
}

# training_dataset = {
#     '3': ['7','8']}

#
# training_dataset = {
#     '1': ['15']}
training = ub.batch_loading(dataset_dir, training_dataset)

validation_dataset = {
    '1': ['21_f'],
#    '3': ['7','11_f']
}

# validation_dataset = {
#     '1': ['15']}
validation = ub.batch_loading(dataset_dir, validation_dataset)

m3=mod.MV3D()

m3.train(max_iter=10000, pre_trained=True,train_set=training,validation_set=validation)
