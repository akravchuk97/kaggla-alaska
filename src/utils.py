import os
from contextlib import suppress

from shutil import copyfile


def copyfile2dir(filepath: str, dir_: str) -> None:
    filename = os.path.basename(filepath)
    copyfile(filepath, os.path.join(dir_, filename))


def write2log(logpath: str, epoch: int, metric: float) -> None:
    add2log = ''
    if epoch == 0:
        add2log = '\t'.join(('epoch', 'metric'))
    add2log += '\t'.join((f'\n{epoch}', str(round(metric, 4))))
    with open(logpath, 'a') as log:
        log.write(add2log)


def make_logdirs_if_needit(logdir: str, experiment_dir: str) -> None:
    with suppress(FileExistsError):
        os.mkdir(logdir)
    with suppress(FileExistsError):
        os.mkdir(os.path.join(logdir, experiment_dir))
    with suppress(FileExistsError):
        os.mkdir(os.path.join(logdir, experiment_dir, 'models'))
    with suppress(FileExistsError):
        os.mkdir(os.path.join(logdir, experiment_dir, 'code'))
    for fn in os.listdir():
        if fn.endswith('.py'):
            copyfile2dir(fn, os.path.join(logdir, experiment_dir, 'code'))
