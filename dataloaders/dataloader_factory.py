from dataloaders import standard_dataloader

dataloader_map = {
    'standard_dataloader': standard_dataloader
}

def get_dataloader(name):
    if name not in dataloader_map:
        raise ValueError('Name of dataloader unknown %s' % name)

    def get_dataloader_fn(image_datasets, batch_size, num_workers, **kwargs):
        return dataloader_map[name].get_dataloader(image_datasets, batch_size, num_workers, **kwargs)

    return get_dataloader_fn
