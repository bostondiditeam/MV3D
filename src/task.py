import os
import time
import argparse

def run_task(command, time_threshold=None):
    print('\n\n start run:%s' %(command))
    delta_time = 0
    # task 1
    try_max = 3
    try_count = 0
    if time_threshold != None:
        while delta_time < time_threshold and try_count <= try_max:
            start_time = time.time()
            os.system(command)
            delta_time = time.time() - start_time
            print('\n\n{} finished ,detal time : {} retry: {}'.format(command,delta_time, try_count))
            time.sleep(2)
            try_count += 1
    else:
        os.system(command)
        time.sleep(2)
#

def train_rpn(tag):

    run_task('python train.py -t "top_view_rpn" -i 600 '
             '-n %s' % (tag))
    for i in range(20):
        run_task('python train.py -w "top_view_rpn" -t "top_view_rpn" -i 600 '
                 ' -n %s -c True' %(tag))
        run_task('python tracking.py -n %s' % (tag))



def train_img_and_fusion(tag):
    run_task('python train.py -w "top_view_rpn" -t "image_feature,fusion" -i 700 '
             '-n %s' % (tag))
    for i in range(10):
        run_task('python train.py -w "top_view_rpn,image_feature,fusion" -t "image_feature,fusion" -i 600 '
                 ' -n %s -c True' %(tag))
        run_task('python tracking.py -n %s_%d -w "%s"' % (tag,i,tag))



def run(tag):

    task_num =1

    if task_num == 0:
        train_rpn(tag)
    elif task_num == 1:
        train_img_and_fusion(tag)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tracking')
    parser.add_argument('-n', '--tag', type=str, nargs='?', default='unknown_tag',
                        help='set log tag')
    args = parser.parse_args()

    print('\n\n{}\n\n'.format(args))
    tag = args.tag
    if tag == 'unknow_tag':
        # tag = input('Enter log tag : ')
        print('\nSet log tag :"%s" ok !!\n' %tag)
    run(tag)