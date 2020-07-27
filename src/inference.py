import typing as tp
from glob import glob

import pandas as pd
import torch
import ttach as tta
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentations import get_validation_augmentation
from dataset import ClassificationDataset, preprocess
from inference_config import config, InferenceConfig
from utils import make_os_settings


class ModelWrapper(nn.Module):
    """Обертка над моделью, которая приводит модели с выходом под классификацию
    к обычной модели с одним выходом, а форвард модели без выхода под классификацию не меняет."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if len(out) == 2:
            return out[0]
        return out


_tta_transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Rotate90(angles=[0, 90, 180, 270]),
    ]
)


def _get_inference_model(config: InferenceConfig) -> nn.Module:
    model = config.model
    model.load_state_dict(torch.load(config.checkpoint_path, map_location='cpu')['st_d'])
    model = ModelWrapper(model)
    model.to('cuda')
    model.eval()
    return tta.ClassificationTTAWrapper(model, _tta_transforms)


def _get_test_df(data_path: str) -> pd.DataFrame:
    dataset = []

    for label, kind in enumerate(['Test']):
        for path in glob(f'{data_path}/Test/*.jpg'):
            dataset.append({
                'kind': kind,
                'image_name': path.split('/')[-1],
                'label': label
            })
    return pd.DataFrame(dataset)


def _get_inference_dataset(config: InferenceConfig) -> ClassificationDataset:
    return ClassificationDataset(
        df=_get_test_df(config.data_path),
        augmentation=get_validation_augmentation(),
        preprocess=preprocess,
        mode='test',
        data_root=config.data_path,
    )


def _predict(loader: DataLoader, model: nn.Module) -> tp.List[float]:
    predicts = []
    for batch in tqdm(loader):
        with torch.no_grad():
            batch_predict = model(batch.to('cuda'))
        batch_predict = 1 - nn.functional.softmax(batch_predict, dim=1).data.cpu().numpy()[:, 0]
        for pred in batch_predict:
            predicts.append(pred)
    return predicts


def _prepare_and_save_submit(submit: pd.DataFrame, config: InferenceConfig):
    submit['Id'] = submit['image_name']
    submit.drop(['kind', 'image_name', 'label'], axis=1, inplace=True)
    submit.to_csv(config.sumbit_name, index=False)


if __name__ == '__main__':
    make_os_settings(config.cuda_num)
    model = _get_inference_model(config)
    dataset = _get_inference_dataset(config)
    loader = DataLoader(dataset, batch_size=config.bs, shuffle=False, num_workers=config.n_work)

    submit: pd.DataFrame = dataset.df
    predicts = _predict(loader, model)
    submit['Label'] = predicts
    _prepare_and_save_submit(submit, config)
