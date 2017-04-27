import model as mod
import glob
from config import *
import utils.batch_loading as ub


dataset_dir = cfg.PREPROCESSED_DATA_SETS_DIR


training_dataset = {
    '1': ['6_f','9_f','15','20'],
    '2': ['3_f','13'],
    '3': ['2_f','4','6','8']}

# training_dataset = {
#     '1': ['6_f','9_f','15','20'],
#                     }
training = ub.batch_loading(dataset_dir, training_dataset)

validation_dataset = {
    '1': ['11','21_f'],
    '2': ['14_f','17'],
    '3': ['7','11_f']
}

# validation_dataset = {
#     '1': ['11','21_f']
# }
validation = ub.batch_loading(dataset_dir, validation_dataset)

m3=mod.MV3D()
m3.train(max_iter=10000, pre_trained=True,train_set=training,validation_set=validation)
