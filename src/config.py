import typing as tp
from dataclasses import dataclass

from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
from torch.optim.optimizer import Optimizer

from metrics import alaska_weighted_auc
from models import enetb2

rlr_kw = {
    'patience': 3,
    'factor': 0.5,
    'mode': 'max',
}

cos_an_kw = {
    'T_max': 5,
    'eta_min': 1e-5,
}


@dataclass
class Config:
    model: nn.Module
    optimizer: type(Optimizer)
    loss: tp.Union[nn.Module, tp.Tuple[nn.Module, nn.Module, nn.Module]]
    scheduler: type(_LRScheduler)
    scheduler_kwargs: tp.Dict[str, tp.Any]
    batch_size: int
    lr: float
    n_epochs: int
    seed: int
    n_work: int
    device: str
    cuda_num: str
    log_name: str
    experiments_root: str
    experiment_name: str
    single_metric: tp.Callable
    data_path: str
    fold_num: int
    df_folds_path: str
    use_qual: bool

config = Config(
    model=enetb2(),
    optimizer=AdamW,
    loss=CrossEntropyLoss(),
    scheduler=CosineAnnealingLR,
    scheduler_kwargs=cos_an_kw,
    batch_size=40,
    lr=5e-4,
    n_epochs=90,
    seed=42,
    n_work=16,
    device='cuda',
    cuda_num='1',
    log_name='log',
    experiment_name='effnet-b2-fullds-adamw-cos-fold0',
    experiments_root='experiments',
    single_metric=alaska_weighted_auc,
    data_path='/home/data/alaska',
    fold_num=0,
    df_folds_path='../df_folds.csv',
    use_qual=False,
)

