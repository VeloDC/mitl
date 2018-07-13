from trainings import standard_training

trainings_map = {
    'standard_training': standard_training
}

def get_training(name):
    if name not in trainings_map:
        raise ValueError('Name of training unknown %s' % name)

    def get_training_fn(model, **kwargs):
        return trainings_map[name].get_training(model, **kwargs)

    return get_training_fn
