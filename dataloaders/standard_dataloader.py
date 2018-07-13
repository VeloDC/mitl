from torch.utils.data import DataLoader

def get_dataloader(image_datasets, batch_size, num_workers):

    dataloaders = {x: DataLoader(image_datasets[x],
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 drop_last=drop_last)
                   for x, shuffle, drop_last in zip(['train', 'val', 'test'],
                                                    [True, False, False],
                                                    [True, False, False])}

    return dataloaders
