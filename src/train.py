import os
from typing import Callable

import cv2
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config, Config
from dataset import get_train_val_datasets
from losses import FocalLoss
from utils import write2log, make_logdirs_if_needit, make_os_settings

qal_loss = FocalLoss()


def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str,
        use_qual: bool = False,
):
    model.train()
    model.to(device)
    pbar = tqdm(enumerate(loader), total=len(loader))
    sum_loss = 0
    for batch_num, batch in pbar:
        if not use_qual:
            im, t0 = batch
        else:
            im, t0, t1 = batch
            t1 = t1.to(device)
        im = im.to(device)
        t0 = t0.to(device)
        if not use_qual:
            pred0 = model(im)
        else:
            pred0, pred_qual = model(im)
        if criterion.__class__.__name__ == 'CrossEntropyLoss':
            loss = criterion(pred0, t0.argmax(1))
        else:
            loss = criterion(pred0, t0)
        if use_qual:
            loss1 = qal_loss(pred_qual, t1)
            loss = loss + loss1
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        sum_loss += loss.item()
        pbar.set_description(f'Loss: {sum_loss / (batch_num + 1)}')

        del loss, pred0, im


def valid_epoch(
        model: nn.Module,
        loader: DataLoader,
        device: str,
        single_metric: Callable,
        criterion: nn.Module,
        use_qual: bool = False,
) -> float:
    model.eval()
    model.to(device)
    preds0 = np.empty(0)
    true0 = np.empty(0)
    pbar = tqdm(enumerate(loader), total=len(loader))
    for batch_num, (im, t0) in pbar:
        im = im.to(device)
        t0 = t0.cpu().data.numpy()[:, 0]
        with torch.no_grad():
            if not use_qual:
                pred0 = model(im)
            else:
                pred0, _ = model(im)
            if criterion.__class__.__name__ == 'CrossEntropyLoss':
                pred0 = 1 - nn.functional.softmax(pred0, dim=1).data.cpu().numpy()[:, 0]
            else:
                pred0 = 1 - pred0.cpu().data.numpy()[:, 0]
        preds0 = np.append(preds0, pred0)
        true0 = np.append(true0, 1 - t0)
    return validate(preds0, true0, single_metric)


def validate(predicts: np.ndarray, labels: np.ndarray, single_metric: Callable) -> float:
    return single_metric(labels, predicts)


def train(
        model: torch.nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        config: Config,
):
    model.to(config.device)
    experiment_path = os.path.join(config.experiments_root, config.experiment_name)
    models_path = os.path.join(experiment_path, 'models')
    log_name = config.experiment_name + '.txt'
    criterion = config.loss
    optimizer = config.optimizer(model.parameters(), lr=config.lr)
    scheduler = config.scheduler(optimizer, **config.scheduler_kwargs)
    max_score = 0
    for epoch in range(config.n_epochs):
        train_epoch(model, train_loader, optimizer, criterion, config.device, use_qual=config.use_qual)
        metric = valid_epoch(model, valid_loader, config.device,
                             config.single_metric, criterion, use_qual=config.use_qual)
        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        else:
            scheduler.step(metric)
        if metric > max_score:
            max_score = metric
            model.eval()
            torch.save({'st_d': model.state_dict()}, os.path.join(models_path, 'best.pth'))
        write2log(os.path.join(experiment_path, log_name), epoch, metric,
                  optimizer.state_dict()['param_groups'][0]['lr'])


if __name__ == '__main__':
    torch.set_num_threads(config.n_work)
    cv2.setNumThreads(0)
    make_os_settings(config.cuda_num)
    make_logdirs_if_needit(config.experiments_root, config.experiment_name)
    train_dataset, valid_dataset = get_train_val_datasets(config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.n_work)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.n_work)

    train(config.model, train_loader, valid_loader, config)
