import albumentations as albu


def get_training_augmentation():
    train_transform = [
        albu.RandomRotate90(p=1),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5)
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
    ]
    return albu.Compose(test_transform)
