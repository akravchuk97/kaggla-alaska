import os

from shutil import copyfile


def write2log(logpath: str, epoch: int, metric: float, lr: float):
    add2log = ''
    if epoch == 0:
        add2log = '\t'.join(('epoch', 'metric', 'lr'))
    add2log += '\t'.join((f'\n{epoch}', str(round(metric, 4)), str(round(lr, 7))))
    with open(logpath, 'a') as log:
        log.write(add2log)


def make_os_settings(cuda_num: str):
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_num
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def make_logdirs_if_needit(logdir: str, experiment_dir: str):
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir, experiment_dir), exist_ok=True)
    os.makedirs(os.path.join(logdir, experiment_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(logdir, experiment_dir, 'code'), exist_ok=True)
    for fn in os.listdir():
        if fn.endswith('.py'):
            copyfile2dir(fn, os.path.join(logdir, experiment_dir, 'code'))


def copyfile2dir(filepath: str, dir_: str):
    filename = os.path.basename(filepath)
    copyfile(filepath, os.path.join(dir_, filename))
