import os
import time
import config

def copy_weigths(dir):
    print('copy_weigths ...')
    weigth_path=os.path.join(config.cfg.CHECKPOINT_DIR,'*')
    os.makedirs(dir, exist_ok=True)
    os.system('cp {} {}'.format(weigth_path, dir))
    os.system('cp {} {}'.format(os.path.join(config.cfg.CHECKPOINT_DIR,'checkpoint'), dir))
    print('copy weigths done')

def task1():
    """
    train image_feature and fusion net (use rpn net pretrained weights )
    :return:
    """
    delta_time=0

    #task 1
    try_max=3
    try_count=0
    while delta_time<120 and try_count<=try_max:
        start_time=time.time()
        os.system('python train.py -w "top_view_rpn" -t "image_feature,fusion" -i 2000')
        delta_time=time.time()-start_time
        print('\n\ntraining finished ,detal time : {} retry: {}'.format(delta_time,try_count) )
        time.sleep(2)
        os.system('python tracking.py')
        time.sleep(2)
        try_count += 1
#

def train_rpn():
    """
    Only train top_view_rpn net (not use pretrained weights )
    :return:
    """
    delta_time=0

    #task 1
    try_max=3
    try_count=0
    while delta_time<120 and try_count<=try_max:
        start_time=time.time()
        os.system('python train.py -t "top_view_rpn" -i 2000')
        time.sleep(2)
        delta_time=time.time()-start_time
        print('\n\ntraining finished ,detal time : {} retry: {}'.format(delta_time,try_count) )

        os.system('python tracking.py')
        time.sleep(2)
        try_count += 1

def main():

    task_num =1

    if task_num == 0:
        # tain all
        train_rpn()
        for i in range(20):
            task1()
            time.sleep(5)
    elif task_num == 1:
        # train imfeature and fusion net
        for i in range(20):
            task1()
            time.sleep(5)

if __name__ == '__main__':
    main()