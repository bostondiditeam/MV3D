import os
import time

def main():
    delta_time=0

    #task 1
    try_max=3
    try_count=0
    while delta_time<3600 and try_count<=try_max:
        start_time=time.time()
        os.system('python train.py')
        delta_time=time.time()-start_time
        print('\n\ntraining finished ,detal time : {} retry: {}'.format(delta_time,try_count) )
        try_count += 1

    #task 2
    os.system('python data.py')
if __name__ == '__main__':
    main()