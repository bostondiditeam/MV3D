import os
import time
import config

def copy_weigths(dir):
    print('copy_weigths ...')
    weigth_path=os.path.join(config.cfg.CHECKPOINT_DIR,'*')
    os.makedirs(dir, exist_ok=True)
    os.system('cp {} {} -r'.format(weigth_path, dir))
    print('copy weigths done')



def run_task(command, time_threshold=120):

    delta_time = 0

    # task 1
    try_max = 3
    try_count = 0
    while delta_time < time_threshold and try_count <= try_max:
        start_time = time.time()
        os.system(command)
        delta_time = time.time() - start_time
        print('\n\n{} finished ,detal time : {} retry: {}'.format(command,delta_time, try_count))
        time.sleep(2)
        try_count += 1
#

def train_rpn():

    run_task('python train.py -t "top_view_rpn" -i 600')
    for i in range(20):
        run_task('python train.py -w "top_view_rpn" -t "top_view_rpn" -i 600')
        run_task('python tracking.py',time_threshold=10)



def train_img_and_fusion():
    run_task('python train.py -w "top_view_rpn" -t "image_feature,fusion" -i 700')
    for i in range(20):
        run_task('python train.py -w "top_view_rpn,image_feature,fusion" -t "image_feature,fusion" -i 600')
        run_task('python tracking.py',time_threshold=10)



def main():

    task_num =1

    if task_num == 0:
        train_rpn()
    elif task_num == 1:
        train_img_and_fusion()

if __name__ == '__main__':
    main()