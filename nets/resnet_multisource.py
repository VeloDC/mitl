import torch
import torch.nn as nn
from torch.nn import ModuleList as ML
from torch.nn import ParameterList as PL

from mitl_layers import Filter_Sum_Forward
from mitl_layers import MaskedConv2d

from torchvision.models import resnet


__all__ = ['resnet18_multisource']  


def resnet18_multisource(nets=[], num_classes=None, masking_fn=lambda x: x):
    main_branch = resnet.resnet18(pretrained=True)
    full_model = ResNet18_MultiSource(main_branch, nets, num_classes=num_classes, masking_fn=masking_fn)
    return full_model


class BasicBlock_MultiSource(nn.Module):
    def __init__(self, main_block, blocks, masking_fn):
        super(BasicBlock_MultiSource, self).__init__()

        self.conv1 = Filter_Sum_Forward(masking_fn(main_block.conv1), ML([b.conv1 for b in blocks]))
        self.bn1 = main_block.bn1
        self.relu = main_block.relu
        self.conv2 = Filter_Sum_Forward(masking_fn(main_block.conv2), ML([b.conv2 for b in blocks]))
        self.bn2 = main_block.bn2

        if main_block.downsample:
            self.downsample = nn.Sequential(
                Filter_Sum_Forward(masking_fn(main_block.downsample[0]), ML([b.downsample[0] for b in blocks])),
                main_block.downsample[1])

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet18_MultiSource(nn.Module):
    def __init__(self, main_branch, nets, num_classes=None, masking_fn=lambda x: x):
        super(ResNet18_MultiSource, self).__init__()

        self.conv1 = Filter_Sum_Forward(masking_fn(main_branch.conv1), ML([n.conv1 for n in nets]))
        self.bn1 = main_branch.bn1
        self.relu = main_branch.relu
        self.maxpool = main_branch.maxpool
        
        self.layer1 = nn.Sequential(
            BasicBlock_MultiSource(main_branch.layer1[0], ML([n.layer1[0] for n in nets]), masking_fn),
            BasicBlock_MultiSource(main_branch.layer1[1], ML([n.layer1[1] for n in nets]), masking_fn))
        self.layer2 = nn.Sequential(
            BasicBlock_MultiSource(main_branch.layer2[0], ML([n.layer2[0] for n in nets]), masking_fn),
            BasicBlock_MultiSource(main_branch.layer2[1], ML([n.layer2[1] for n in nets]), masking_fn))
        self.layer3 = nn.Sequential(
            BasicBlock_MultiSource(main_branch.layer3[0], ML([n.layer3[0] for n in nets]), masking_fn),
            BasicBlock_MultiSource(main_branch.layer3[1], ML([n.layer3[1] for n in nets]), masking_fn))
        self.layer4 = nn.Sequential(
            BasicBlock_MultiSource(main_branch.layer4[0], ML([n.layer4[0] for n in nets]), masking_fn),
            BasicBlock_MultiSource(main_branch.layer4[1], ML([n.layer4[1] for n in nets]), masking_fn))
        
        self.avgpool = main_branch.avgpool
        if num_classes is not None:
            self.fc = nn.Linear(512, num_classes)
        else:
            self.fc = main_branch.fc

    def forward(self, x):        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
