import model as mod
import glob
from config import *
import utils.batch_loading as ub
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('-w', '--weights', type=str, nargs='?', default='all',
        help='use pre trained weigthts example: -w "rpn,fusion" ')
    parser.add_argument('-i', '--max_iter', type=int, nargs='?', default=1000,
                        help='max count of train iter')
    args = parser.parse_args()

    print('\n\n{}\n\n'.format(args))
    max_iter = args.max_iter
    weights = args.weights.split(',')

    dataset_dir = cfg.PREPROCESSED_DATA_SETS_DIR

    # ```training_dataset = {
    #     '1': ['6_f','9_f','10', '13', '20'],
    #     '2': ['3_f', '6_f', '8_f'],
    #     '3': ['2_f','4','6','8']}```
    #
    # new messages
    # [3:18]
    #  ```validation_dataset = {
    #     '1': ['21_f', '15', '19'],
    #     '3': ['7','11_f']
    # }```

    # training_dataset = {
    #     '1': ['6_f','9_f','10', '13', '20'],
    #     '2': ['3_f', '6_f', '8_f'],
    #     '3': ['2_f','4','6','8']}

    training_dataset = {
        '1': ['6_f', '9_f', '10', '13', '20', '21_f', '15', '19'],
        '2': ['3_f', '6_f', '8_f'],
        '3': ['2_f', '4', '6', '8', '7', '11_f']}

    #
    # training_dataset = {
    #     '1': ['15']}
    training = ub.batch_loading(dataset_dir, training_dataset)

    validation_dataset = {
        '1': ['21_f', '15', '19'],
        '3': ['7', '11_f']
    }

    # validation_dataset = {
    #     '1': ['15']}
    validation = ub.batch_loading(dataset_dir, validation_dataset)

    m3 = mod.MV3D()

    m3.train(max_iter=max_iter, pre_trained_weights=weights,
             train_set=training, validation_set=validation)


