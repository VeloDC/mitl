import torch
from torch.autograd import Variable
from torchvision.models import resnet
from resnet_multisource import resnet18_multisource

from mitl_layers import MaskedConv2d
from mitl_layers import QuantizedConv2d


def test_resnet18_multisource(nets, masking_fn, bs=8):
    model = resnet18_multisource(nets, masking_fn=masking_fn)
    print model
    inputs = torch.ones(bs, 3, 224, 224)
    inputs = torch.autograd.Variable(inputs)
    o = model(inputs)
    return True


def main():
    nets = [resnet.resnet18()]

    print('Testing no mask model')
    test_resnet18_multisource(nets, masking_fn=lambda x: x)
    print('Testing Piggyback mask model')
    test_resnet18_multisource(nets, masking_fn=MaskedConv2d)
    print('Testing Quantized mask model')
    test_resnet18_multisource(nets, masking_fn=QuantizedConv2d)

if __name__=='__main__':
    main()
