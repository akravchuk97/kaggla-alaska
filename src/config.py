import typing as tp

from dataclasses import dataclass
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim.optimizer import Optimizer
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, CosineAnnealingLR
from torch.nn.modules.loss import BCEWithLogitsLoss
from models import get_model, enetb2_cls, enetb3_cls, enetb2, enetb3
from metrics import alaska_weighted_auc
from losses import FocalLoss
from adam_gcc import Adam_GCC2, AdamW_GCC2

rlr_kw = {
    'patience': 3,
    'factor': 0.5,
    'mode': 'max',
    'verbose': True,
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
    use_sm_b: bool
    use_qual: bool
    accum_steps: int


config = Config(
    model=enetb2(),
    optimizer=AdamW,
    loss=CrossEntropyLoss(),
    scheduler=CosineAnnealingLR,
    scheduler_kwargs=cos_an_kw,
    batch_size=40,
    lr=5e-4,
    n_epochs=300,
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
    use_sm_b=False,
    use_qual=False, 
    accum_steps=1,
)

