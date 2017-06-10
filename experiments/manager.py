import os
import glob
import subprocess

class Env(object):
    def __init__(self,root_dir:str, dep_links:dict, dep_copy:dict =None):
        self.root_dir=root_dir
        self.dep_links=dep_links
        self.dep_copy = dep_copy

        #reset dependence links
        print('reset dep links')
        self.unset_dep_links()
        self.set_dep_links()


    def set_dep_links(self):
        for k in self.dep_links:
            src_path = os.path.join(self.root_dir, k)
            des_path = os.path.join(self.root_dir, self.dep_links[k])
            command = 'ln -s %s %s' %(des_path, src_path)
            os.system(command)
            # print('run: ',command)


    def unset_dep_links(self):
        for k in self.dep_links:
            path = os.path.join(self.root_dir, k)
            command = 'unlink %s' %(path)
            if os.path.islink(path):
                state = os.system(command)
                # print('run: ',command)


class Experimet(Env):
    def __init__(self, dir, dep_links):
        Env.__init__(self, dir, dep_links=dep_links)
        self.tag = os.path.basename(dir)
        self.dir = dir
        self.check_state = 'unknown'
        self.run_state = 'unknown'


    def run(self):
        command = 'cd %s && python task.py -n %s' % (self.dir,self.tag)
        print('\nrun: %s\n' % (command))
        code = subprocess.call(command, shell=True)
        if code != 0:
            self.run_state = 'fail'
            return 1
        else:
            self.run_state = 'success'
            return 0

    def check(self):
        command = 'cd %s && python task.py -n %s -t True' % (self.dir,self.tag)
        print('\nrun: %s\n' % (command))
        code = subprocess.call(command, shell=True)
        if code != 0:
            self.check_state = 'fail'
            return 1
        else:
            self.check_state = 'success'
            return 0


class Manager(Env):
    def __init__(self, root_dir=os.path.abspath('./')):
        links={
            'log':'../log',
            'checkpoint': '../checkpoint',
            'data': '../data',
        }
        Env.__init__(self, root_dir=root_dir, dep_links=links)


    def scan(self):
        dirs = glob.glob(os.path.join(self.root_dir,'exp_*'))
        dirs.sort()

        dep_links={
            'net':os.path.join(self.root_dir,'..','src','net'),
            'didi_data': os.path.join(self.root_dir, '..', 'src', 'didi_data'),
            'kitti_data': os.path.join(self.root_dir, '..', 'src', 'kitti_data'),
            'tracklets': os.path.join(self.root_dir, '..', 'src', 'tracklets'),
            'data.py': os.path.join(self.root_dir, '..', 'src', 'data.py'),
            'utils': os.path.join(self.root_dir, '..', 'src', 'utils'),
            'train.py': os.path.join(self.root_dir, '..', 'src', 'train.py'),
            'tracking.py': os.path.join(self.root_dir, '..', 'src', 'tracking.py'),
            'task.py': os.path.join(self.root_dir, '..', 'src', 'task.py'),
        }
        return [Experimet(dir=dir,dep_links=dep_links) for dir in dirs]

    def summary(self, exps):
        print('\n------------------------------------------------------------------------------')
        print('\nexperiments has :')
        for i, exp in enumerate(exps):
            print('    %d: %s' % (i, exp.tag))

    def check(self, exps: [Experimet]):
        sum = 0
        # check
        for i, exp in enumerate(exps):
            print('\n\n-----------------------experiment check: %d/%d -------------------' %
                  (i + 1, len(exps)))
            sum += exp.check()

        print('\n------------------------------------------------------------------------------')
        for i, exp in enumerate(exps):
            print('fast test : %d. %s  %s' %(i,exp.tag, exp.check_state))

        return sum

    def run(self, exps: [Experimet]):
        for i,exp in enumerate(exps):
            print('\n\n-----------------------experiment run: %d/%d -------------------' %
                  (i+1,len(exps)))
            exp.run()

        print('\n------------------------------------------------------------------------------')
        for i, exp in enumerate(exps):
            print('exp run : %d. %s  %s' %(i,exp.tag, exp.run_state))




if __name__ == '__main__':
    man = Manager()
    print('\n\n start scan all experiments')
    exps = man.scan()

    man.summary(exps)

    fail_sum = man.check(exps)

    if fail_sum == 0:
        man.run(exps)
