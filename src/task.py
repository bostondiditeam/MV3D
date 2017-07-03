import os
import time
import argparse
import subprocess


def run_task(command, time_threshold=None):
    print('\nStart run:\n"%s"\n' % (command))
    delta_time = 0
    # task 1
    try_max = 3
    try_count = 0
    if time_threshold != None:
        while delta_time < time_threshold and try_count <= try_max:
            start_time = time.time()
            os.system(command)
            delta_time = time.time() - start_time
            print('\n\n{} finished ,detal time : {} retry: {}'.format(command, delta_time, try_count))
            time.sleep(2)
            try_count += 1
    else:
        exit_code = subprocess.call(command, shell=True)
        if exit_code != 0: exit(exit_code)

class Task(object):

    def __init__(self, fast_test=False, tag ='unknown_tag'):
        self.fast_test = fast_test
        self.tag=tag

    def train_rpn(self):
        iter = lambda i: i if self.fast_test == False else 1

        run_task('python train.py -w "" -t "top_view_rpn" -i %d '
                 '-n %s' % (iter(5000), self.tag))
        run_task('python train.py -w "top_view_rpn" -t "image_feature,fusion" -i %d '
                 '-n %s -c True' % (iter(200), self.tag))
        run_task('python tracking.py -n %s_%d -w "%s" -t %s -s 100' % (tag, 100, tag, self.fast_test))

        for i in range(iter(5)):
            run_task('python train.py -w "top_view_rpn" -t "top_view_rpn" -i %d '
                     ' -n %s -c True' % (iter(3000), tag))
            run_task('python tracking.py -n %s_%d -w "%s" -t %s -s 100' % (tag, i, tag, self.fast_test))



    def train_img_and_fusion(self, continue_train=False, init_train=4000,tracking_max=3,
                             train_epoch=2500,tracking_skip_frames=0):

        iter = lambda i:  i if self.fast_test==False else 1

        if continue_train == False:
            run_task('python train.py -w "top_view_rpn" -t "image_feature,fusion" -i %d '
                     '-n %s' % (iter(init_train), self.tag))

        for i in range(iter(tracking_max)):
            run_task('python train.py -w "top_view_rpn,image_feature,fusion" -t "image_feature,fusion" -i %d '
                     ' -n %s -c True' %(iter(train_epoch), tag))
            run_task('python tracking.py -n %s_%d -w "%s" -t %s -s %d' % (tag,i,tag,self.fast_test,
                                                                          tracking_skip_frames))

    def tracking(self, tracking_skip_frames=0,weights=None):
        weights = self.tag if weights==None else weights
        run_task('python tracking.py -n %s -w "%s" -t %s -s %d' % (tag, weights ,self.fast_test,
                                                                          tracking_skip_frames))

    def banchmark(self, tracking_skip_frames=100, tracking_range=range(1)):
        tracking_range= tracking_range if self.fast_test==False else range(1)
        for i in tracking_range:
            run_task('python tracking.py -n benchmark_%s_%d -w "%s_%d" -t %s -s %d' %
                     (tag,i, tag,i ,self.fast_test,tracking_skip_frames))


def str2bool(v: str):
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tracking')
    parser.add_argument('-n', '--tag', type=str, nargs='?', default='unknown_tag',
                        help='set log tag')
    parser.add_argument('-t', '--fast_test', type=str2bool, nargs='?', default=False,
                        help='fast test mode')
    args = parser.parse_args()

    print('\n\n{}\n\n'.format(args))
    fast_test = bool(args.fast_test)
    tag = args.tag
    if tag == 'unknow_tag':
        tag = input('Enter log tag : ')
        print('\nSet log tag :"%s" ok !!\n' %tag)

    # Task(tag=tag, fast_test=args.fast_test).train_img_and_fusion(init_train=3000,
    #                                                              train_epoch=5000,tracking_max=4)
    #
    Task(tag=tag, fast_test=args.fast_test).train_rpn()