import typing as tp

from dataclasses import dataclass
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim.optimizer import Optimizer
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.nn.modules.loss import BCEWithLogitsLoss
from models import get_model, enetb2_cls, enetb3_cls
from metrics import alaska_weighted_auc
from losses import FocalLoss
from adam_gcc import Adam_GCC2, AdamW_GCC2

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
    use_sm_b: bool
    use_qual: bool


config = Config(
    model=enetb3_cls(idx=12),
    optimizer=Adam_GCC2,
    loss=CrossEntropyLoss(),
    scheduler=ReduceLROnPlateau,
    scheduler_kwargs=rlr_kw,
    batch_size=32,
    lr=5e-4,
    n_epochs=150,
    seed=42,
    n_work=12,
    device='cuda',
    cuda_num='6',
    log_name='log',
    experiment_name='effnet-b3-qual-12-ce-gcc-fullds',
    experiments_root='experiments',
    single_metric=alaska_weighted_auc,
    data_path='/home/data/alaska',
    fold_num=6,
    df_folds_path='../folds_with_qual.csv',
    use_sm_b=False,
    use_qual= True, 
)

