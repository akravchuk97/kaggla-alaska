from dataclasses import dataclass

from torch import nn

from models import enetb2


@dataclass
class InferenceConfig:
    model: nn.Module
    checkpoint_path: str
    cuda_num: str
    n_work: int
    bs: int
    data_path: str
    sumbit_name: str


config = InferenceConfig(
    model=enetb2(),
    checkpoint_path='/path/to/check.pth',
    cuda_num='1',
    n_work=10,
    bs=24,
    data_path='/home/data/alaska',
    sumbit_name='submit.csv',
)
