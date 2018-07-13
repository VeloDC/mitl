from torchvision.models import alexnet
import torch.nn as nn


def get_network(num_classes, pretrained=False):
    net = alexnet(pretrained=pretrained)
    model = AlexNet(net, num_classes)
    return model


class AlexNet(nn.Module):
    def __init__(self, net, num_classes):
        super(AlexNet, self).__init__()
        
        self.conv1 = nn.Sequential(*list(net.features.children())[0:3])
        self.conv2 = nn.Sequential(*list(net.features.children())[3:6])
        self.conv3 = nn.Sequential(*list(net.features.children())[6:8])
        self.conv4 = nn.Sequential(*list(net.features.children())[8:10])
        self.conv5 = nn.Sequential(*list(net.features.children())[10:13])
        self.fc6 = nn.Sequential(*list(net.classifier.children())[0:3])
        self.fc7 = nn.Sequential(*list(net.classifier.children())[3:6])
        self.classifier = nn.Sequential(nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = x.view(x.size(0), 256 * 6 * 6)

        x = self.fc6(x)
        x = self.fc7(x)
        x = self.classifier(x)
        
        return x
