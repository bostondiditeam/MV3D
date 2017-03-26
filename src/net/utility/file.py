import builtins as __builtin__
import sys
import os
import glob
import shutil


### http://stackoverflow.com/questions/1706198/python-how-to-ignore-comment-lines-when-reading-in-a-file
### http://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do

# naming conventions
#   e.g. c:/xxx/yyy/zzz.txt
#
#    filename = zzz.txt
#    file or filepath = c:/xxx/yyy/zzz.txt
#
#   e.g. c:/xxx/yyy/zzz
#
#    dirname = zzz
#    dir or dirpath = c:/xxx/yyy/zzz
#
#

def remove_comments(lines, token='#'):
    """ Generator. Strips comments and whitespace from input lines.
    """

    l = []
    for line in lines:
        s = line.split(token, 1)[0].strip()
        if s != '':
            l.append(s)
    return l


def open(file, mode=None, encoding=None):
    if mode == None: mode = 'r'

    if '/' in file:
        if 'w' or 'a' in mode:
            dir = os.path.dirname(file)
            if not os.path.isdir(dir):  os.makedirs(dir)

    f = __builtin__.open(file, mode=mode, encoding=encoding)
    return f


def makedirs(dir):
    if not os.path.isdir(dir): os.makedirs(dir)


def remove(file):
    if os.path.exists(file): os.remove(file)


def empty(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir, ignore_errors=True)
    else:
        os.makedirs(dir)


class Logger(object):
    def __init__(self,file=None, mode=None):
        self.terminal = sys.stdout
        self.file = None
        if file is not None: self.open(file,mode)


    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if message =='\r': is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


##sys.stdout = Logger()

# main --------------------------------------------------------------------------
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))