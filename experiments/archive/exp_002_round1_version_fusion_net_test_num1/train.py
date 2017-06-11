import mv3d
import mv3d_net
import glob
from config import *
import utils.batch_loading as ub
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')

    all= '%s,%s,%s' % (mv3d_net.top_view_rpn_name ,mv3d_net.imfeature_net_name,mv3d_net.fusion_net_name)

    parser.add_argument('-w', '--weights', type=str, nargs='?', default='',
        help='use pre trained weigthts example: -w "%s" ' % (all))

    parser.add_argument('-t', '--targets', type=str, nargs='?', default=all,
        help='train targets example: -w "%s" ' % (all))

    parser.add_argument('-i', '--max_iter', type=int, nargs='?', default=1000,
                        help='max count of train iter')

    parser.add_argument('-n', '--tag', type=str, nargs='?', default='unknown_tag',
                        help='set log tag')

    parser.add_argument('-c', '--continue_train', type=bool, nargs='?', default=False,
                        help='set continue train flag')
    args = parser.parse_args()

    print('\n\n{}\n\n'.format(args))
    tag = args.tag
    if tag == 'unknown_tag':
        tag = input('Enter log tag : ')
        print('\nSet log tag :"%s" ok !!\n' %tag)

    max_iter = args.max_iter
    weights=[]
    if args.weights != '':
        weights = args.weights.split(',')

    targets=[]
    if args.targets != '':
        targets = args.targets.split(',')

    dataset_dir = cfg.PREPROCESSED_DATA_SETS_DIR

    if cfg.DATA_SETS_TYPE == 'didi' or cfg.DATA_SETS_TYPE == 'test':
        training_dataset = {
            '1': ['6_f', '9_f', '10', '13', '20', '21_f', '15', '19'],
            '2': ['3_f', '6_f', '8_f'],
            '3': ['2_f', '4', '6', '8', '7', '11_f']}

        validation_dataset = {
            '1': ['15']}

    elif cfg.DATA_SETS_TYPE == 'kitti':
        training_dataset = {
            '2011_09_26': ['0001', '0017', '0029', '0052', '0070', '0002', '0018', '0056',  '0019',
                       '0036', '0005',
                       '0057', '0084', '0020', '0039', '0086', '0011', '0023', '0046', '0060', '0091']}

        validation_dataset = {
            '2011_09_26': ['0013', '0027', '0048',
                           '0061', '0015', '0028', '0051', '0064']
        }

    training = ub.batch_loading(dataset_dir, training_dataset)

    validation = ub.batch_loading(dataset_dir, validation_dataset)

    train = mv3d.Trainer(train_set=training, validation_set=validation,
                         pre_trained_weights=weights, train_targets=targets, log_tag=tag,
                         continue_train = args.continue_train)

    train(max_iter=max_iter)


