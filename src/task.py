import os
import time
import config

def copy_weigths(dir):
    print('copy_weigths ...')
    weigth_path=os.path.join(config.cfg.CHECKPOINT_DIR,'*.*')
    os.makedirs(dir, exist_ok=True)
    os.system('cp {} {}'.format(weigth_path, dir))
    os.system('cp {} {}'.format(os.path.join(config.cfg.CHECKPOINT_DIR,'checkpoint'), dir))
    print('copy weigths done')

def task1():
    delta_time=0

    #task 1
    try_max=3
    try_count=0
    while delta_time<600 and try_count<=try_max:
        start_time=time.time()
        os.system('python train.py')
        delta_time=time.time()-start_time
        print('\n\ntraining finished ,detal time : {} retry: {}'.format(delta_time,try_count) )

        os.system('python tracking.py')
        try_count += 1

def main():
    for i in range(10):
        task1()
        time.sleep(5)

if __name__ == '__main__':
    main()