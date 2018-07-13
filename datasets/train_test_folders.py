import os
from torchvision import datasets


def make_train_val_splits(dataset_dir):
    target = os.path.join(dataset_dir, 'pytorch')
    os.mkdir(target)
    for d in ['train', 'val']:
        os.mkdir(os.path.join(target, d))

    for label in os.listdir(os.path.join(dataset_dir, 'train')):

        for d in ['train', 'val']:
            os.mkdir(os.path.join(target, d, label))
        
        filenames = os.listdir(os.path.join(dataset_dir, 'train', label))
        filenames = map(lambda x: os.path.join(dataset_dir, 'train', label, x), filenames)

        for i, item in enumerate(filenames):
            if i % 10 == 0:
                os.symlink(item, os.path.join(target, 'val', label, item.split('/')[-1]))

            else:
                os.symlink(item, os.path.join(target, 'train', label, item.split('/')[-1]))


def get_dataset(dataset_dir, data_transforms):

    if not os.path.exists(os.path.join(dataset_dir, 'pytorch')):
        make_train_val_splits(dataset_dir)
        os.symlink(os.path.join(dataset_dir, 'test'), os.path.join(dataset_dir, 'pytorch', 'test'))
        
    image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_dir, 'pytorch', x), data_transforms[x])
                      for x in ['train', 'val', 'test']}

    return image_datasets
