from time import time

class timer:
    def __init__(self):
        self.init_time = time()
        self.time_now = self.init_time

    def time_diff_per_n_loops(self):
        time_diff = time() - self.time_now
        self.time_now = time()
        return time_diff

    def total_time(self):
        return time() - self.init_time



if __name__ == '__main__':
    from time import sleep

    timeit = timer()
    for i in range(10):
        sleep(1)
        print('It takes {} secs per loop.'.format(timeit.time_diff_per_n_loops()))

    print('It takes {} secs per whole script.'.format(timeit.total_time()))

