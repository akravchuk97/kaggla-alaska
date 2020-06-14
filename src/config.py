import typing as tp

from dataclasses import dataclass
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.nn.modules.loss import BCEWithLogitsLoss
from models import get_model
from metrics import alaska_weighted_auc

rlr_kw = {
    'patience': 3,
    'factor': 0.5,
    'mode': 'max',
    'verbose': True,
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


config = Config(
    model=get_model(n_classes=4),
    optimizer=Adam,
    loss=BCEWithLogitsLoss(),
    scheduler=ReduceLROnPlateau,
    scheduler_kwargs=rlr_kw,
    batch_size=2,
    lr=5e-4,
    n_epochs=150,
    seed=42,
    n_work=10,
    device='cuda',
    cuda_num='2',
    log_name='log',
    experiment_name='testim',
    experiments_root='experiments',
    single_metric=alaska_weighted_auc,
    data_path='../alaska',
    fold_num=0,
    df_folds_path='../df_folds.csv'
)

