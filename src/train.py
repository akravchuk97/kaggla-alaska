import os
from typing import Tuple, List, Callable

import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config, Config
from dataset import get_train_val_datasets
from utils import write2log, make_logdirs_if_needit



def validate(predicts, labels, single_metric: Callable):
    return single_metric(labels, predicts)


def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str,
) -> None:
    model.train()
    model.to(device)
    pbar = tqdm(enumerate(loader), total=len(loader))
    sum_loss = 0
    for batch_num, (im, t0) in pbar:
        optimizer.zero_grad()
        im = im.to(device)
        t0 = t0.to(device)
        pred0 = model(im)

        loss = criterion(pred0, t0)
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        pbar.set_description(f'Loss: {sum_loss / (batch_num + 1)}')

        del loss, pred0, im


def valid_epoch(
        model: nn.Module,
        loader: DataLoader,
        device: str,
        single_metric: Callable,
) -> Tuple[float, List[float]]:
    model.eval()
    model.to(device)
    preds0 = np.empty(0)
    true0 = np.empty(0)
    pbar = tqdm(enumerate(loader), total=len(loader))
    for batch_num, (im, t0) in pbar:
        im = im.to(device)
        t0 = t0.cpu().data.numpy()[:, 0]
        with torch.no_grad():
            pred0 = model(im)
            pred0 = pred0.cpu().data.numpy()[:, 0]
        preds0 = np.append(preds0, pred0)
        true0 = np.append(true0, t0)
    return validate(preds0, true0, single_metric)


def train(
        model: torch.nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        config: Config,
) -> None:
    model.to(config.device)
    experiment_path = os.path.join(config.experiments_root, config.experiment_name)
    models_path = os.path.join(experiment_path, 'models')
    log_name = config.experiment_name + '.txt'
    criterion = config.loss
    optimizer = config.optimizer(model.parameters(), lr=config.lr)
    scheduler = config.scheduler(optimizer, **config.scheduler_kwargs)
    max_score = 0

    for epoch in range(config.n_epochs):
        train_epoch(model, train_loader, optimizer, criterion, config.device)
        metric = valid_epoch(model, valid_loader, config.device,
                             config.single_metric)
        scheduler.step(metric)
        if metric > max_score:
            max_score = metric
            model.eval()
            torch.save({'st_d': model.state_dict()}, os.path.join(models_path, 'best.pth'))
        write2log(os.path.join(experiment_path, log_name), epoch, metric)


def make_os_settings(cuda_num: str) -> None:
    os.environ['TORCH_HOME'] = '/home/ds'
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_num
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_num
    make_os_settings(config.cuda_num)
    make_logdirs_if_needit(config.experiments_root, config.experiment_name)
    model = config.model
    train_dataset, valid_dataset = get_train_val_datasets(config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.n_work)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.n_work)

    train(model, train_loader, valid_loader, config, )
