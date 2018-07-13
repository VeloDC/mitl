from nets import alexnet
from nets import resnet_multisource

nets_map = {
    'alexnet': alexnet.get_network,
    'resnet18_multisource': resnet_multisource.resnet18_multisource,
    'resnet50_multisource': resnet_multisource.resnet50_multisource
    }

def get_network(name):

    if name not in nets_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(num_classes, **kwargs):
        return nets_map[name](num_classes, **kwargs)

    return get_network_fn
