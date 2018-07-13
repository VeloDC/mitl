import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ParameterList as PL
from torch.nn.parameter import Parameter


class Filter_Sum_Forward(nn.Module):
    '''
    Args:
        main_conv (Conv2d): La convoluzione del main branch
        side_convs (ModuleList of Conv2d): Le convoluzioni dei branch laterali

    Attributes:
        alphas (`list` of nn.Parameters): Un alpha per ciascuno dei branch laterali

    Output:
        Wmain(x) + SUM(Wside(x)*alpha)
    '''
    def __init__(self, main_conv, side_convs):
        super(Filter_Sum_Forward, self).__init__()

        self.main_conv = main_conv
        self.side_convs = side_convs

        self.alphas = PL([nn.Parameter(torch.zeros(1)) for _ in range(len(self.side_convs))])

    def side_output(self, x, i):
        side_output = self.side_convs[i](x)
        side_output = side_output.mul_(self.alphas[i])
        return side_output

    def forward(self, x):        
        output_main = self.main_conv(x)
        side_outputs = [self.side_output(x, i) for i in range(len(self.side_convs))]
        for o in side_outputs:
            output_main.add_(o)
        return output_main


class MaskedConv2d(nn.modules.conv.Conv2d):
    '''
    Funzione che prende in input una Conv2d e resituisce una MaskedConv2d con i parametri ed i
    pesi della convoluzione in input 

    Args:
        conv (Conv2d): La convoluzione che MaskedConv2d va a mascherare
    '''
    def __init__(self, conv):
        super(MaskedConv2d, self).__init__(conv.in_channels, conv.out_channels, conv.kernel_size,
                                           conv.stride, conv.padding, conv.dilation,
                                           conv.groups, conv.bias)

        self.weight = conv.weight
        self.weight.requires_grad = False
        if conv.bias:
            self.bias = conv.bias
            self.bias.requires_grad = False
        self.treshold = 0.

        self.mask = Parameter(torch.Tensor(self.out_channels, self.in_channels // self.groups,
                                           *self.kernel_size))
        self.treshold_mask = 0
        self.reset_mask()

    def reset_mask(self):
        self.mask.data.fill_(0.01)

    def forward(self, x):
        binary_mask = self.mask.clone()
        binary_mask.data = (binary_mask.data > self.treshold).float()
        W = binary_mask * self.weight

        return F.conv2d(x, W, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuantizedConv2d(nn.modules.conv.Conv2d):
    '''
    (come MaskedConv2d)
    '''
    def __init__(self, conv):
        super(QuantizedConv2d, self).__init__(conv.in_channels, conv.out_channels, conv.kernel_size,
                                              conv.stride, conv.padding, conv.dilation,
                                              conv.groups, conv.bias)
        self.weight = conv.weight
        self.weight.requires_grad=False

        if conv.bias:
            self.bias = conv.bias
            self.bias.requires_grad = False

        self.n_masks = 1
        self.additional_mask = Parameter(torch.Tensor(self.out_channels, self.in_channels // self.groups,
                                                      *self.kernel_size))

        self.threshold = 0.

        self.bias_mask=Parameter(torch.FloatTensor(1))
        self.scale_mask=Parameter(torch.Tensor(self.n_masks).view(-1,1,1,1))
        self.scale_mask2=Parameter(torch.Tensor(self.n_masks).view(-1,1,1,1))
	
        self.reset_mask()

    def reset_mask(self):
        self.additional_mask.data.uniform_(0.00001,0.00002)
        self.bias_mask.data.fill_(0.0)
        self.scale_mask.data.fill_(0.0)
        self.scale_mask2.data.fill_(0.0)

    def forward(self, x):

        binary_addition_masks=self.additional_mask.clone()
        binary_addition_masks.data=(binary_addition_masks.data>self.threshold).float()

        #W= W_pretrained + a*M + b*W*M + c
        W = self.weight+self.scale_mask*binary_addition_masks+self.scale_mask2*self.weight*binary_addition_masks+self.bias_mask

        return F.conv2d(x, W, self.bias, self.stride, self.padding, self.dilation, self.groups)
