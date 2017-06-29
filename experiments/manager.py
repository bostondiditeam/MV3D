import os
import glob
import subprocess
import time

class Env(object):
    def __init__(self, env_dir:str, dep_links:dict= {}, dep_copys:dict= {}):
        self.env_dir=env_dir
        self.dep_links=dep_links
        self.dep_copys = dep_copys

        #reset dependence links
        print('reset dep links')
        self.set_dep_links()
        self.set_dep_copys()


    def set_dep_links(self):
        for k in self.dep_links:
            src_path = os.path.join(self.env_dir, k)
            des_path = os.path.join(self.env_dir, self.dep_links[k])
            command = 'ln -s %s %s' %(des_path, src_path)
            code = subprocess.call(command, shell=True)
            if code != 0: exit(code)
            # print('run: ',command)


    def set_dep_copys(self):
        for k in self.dep_copys:
            src_path = os.path.join(self.env_dir, k)
            des_path = os.path.join(self.env_dir, self.dep_copys[k])
            if os.path.exists(src_path)==False:
                command = 'cp -r %s %s' %(des_path, src_path)
                code = subprocess.call(command, shell=True)
                if code != 0: exit(code)
            # print('run: ',command)


    def unset_dep_copys(self):
        for k in self.dep_copys:
            path = os.path.join(self.env_dir, k)
            if os.path.exists(path):
                if os.path.islink(path):
                    command = 'unlink %s' % (path)
                else:
                    command = 'rm -rf %s' % (path)
                code = subprocess.call(command, shell=True)
                if code != 0: exit(code)
                # print('run: ',command)


class Experimet(Env):
    def __init__(self, env_dir, dep_links:dict={}, dep_copys:dict={}):
        Env.__init__(self, env_dir, dep_links=dep_links, dep_copys=dep_copys)
        self.tag = os.path.basename(env_dir)
        self.dir = env_dir
        self.check_state = 'unknown'
        self.run_state = 'unknown'


    def run(self):
        command = 'cd %s && python task.py -n %s' % (self.dir,self.tag)
        print('\nrun: \n"%s"\n' % (command))
        code = subprocess.call(command, shell=True)
        if code != 0:
            self.run_state = 'fail'
            return 1
        else:
            self.run_state = 'success'
            return 0

    def check(self):
        command = 'cd %s && python task.py -n %s -t True' % (self.dir,self.tag)
        print('\nRun: \n"%s"\n' % (command))
        code = subprocess.call(command, shell=True)
        if code != 0:
            self.check_state = 'fail'
            return 1
        else:
            self.check_state = 'success'
            return 0


class Manager(Env):
    def __init__(self):
        self.root_dir=os.path.join(os.path.abspath('./'))
        self.env_dir =os.path.join(self.root_dir, 'temp')
        if os.path.exists(self.env_dir):
            code= subprocess.call('rm -rf %s' % self.env_dir, shell=True)
            if code != 0: exit(code)
        os.makedirs(self.env_dir, exist_ok=True)

        links={
            'log': os.path.join(self.root_dir, '../log'),
            'checkpoint': os.path.join(self.root_dir, '../checkpoint'),
            'data': os.path.join(self.root_dir,'../data'),
        }
        Env.__init__(self, env_dir=self.env_dir, dep_links=links)


    def scan(self):
        dirs = glob.glob(os.path.join(self.root_dir,'exp_*'))
        dirs.sort()
        return dirs


    def creat_env(self, exps_src_dir=[]):

        dep_copys={
            'didi_data': os.path.join(self.root_dir, '..', 'src', 'didi_data'),
            'kitti_data': os.path.join(self.root_dir, '..', 'src', 'kitti_data'),
            'lidar_data_preprocess': os.path.join(self.root_dir, '..', 'src', 'lidar_data_preprocess'),
            'tracklets': os.path.join(self.root_dir, '..', 'src', 'tracklets'),
            'utils': os.path.join(self.root_dir, '..', 'src', 'utils'),
            'net': os.path.join(self.root_dir, '..', 'src', 'net'),
            'data.py': os.path.join(self.root_dir, '..', 'src', 'data.py'),
            'config.py': os.path.join(self.root_dir, '..', 'src', 'config.py'),
            'mv3d.py': os.path.join(self.root_dir, '..', 'src', 'mv3d.py'),
            'mv3d_net.py': os.path.join(self.root_dir, '..', 'src', 'mv3d_net.py'),
            'raw_data.py': os.path.join(self.root_dir, '..', 'src', 'raw_data.py'),
            'train.py': os.path.join(self.root_dir, '..', 'src', 'train.py'),
            'tracking.py': os.path.join(self.root_dir, '..', 'src', 'tracking.py'),
            'task.py': os.path.join(self.root_dir, '..', 'src', 'task.py'),
        }
        exps = []
        for exp_dir in exps_src_dir:
            for files_dir in glob.glob(os.path.join(exp_dir, '*')) :
                dep_copys[os.path.basename(files_dir)] = files_dir

            exp_env_dir = os.path.join(self.env_dir, os.path.basename(exp_dir))
            os.makedirs(exp_env_dir, exist_ok=True)
            exps += [Experimet(env_dir=exp_env_dir, dep_copys=dep_copys)]

        return exps


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
    exps = man.creat_env(exps_src_dir=man.scan())

    man.summary(exps)

    fail_sum = man.check(exps)

    if fail_sum  == 0 and len(exps)!=0:
        print('After 10s will start run all experiments')
        time.sleep(10)
        man.run(exps)
