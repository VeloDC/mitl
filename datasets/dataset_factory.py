from datasets import train_test_folders

datasets_map = {
    'flowers': train_test_folders,
    'cubs': train_test_folders
}

def get_dataset(name):
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)

    def get_dataset_fn(dataset_dir, **kwargs):
        return datasets_map[name].get_dataset(dataset_dir, **kwargs)

    return get_dataset_fn
