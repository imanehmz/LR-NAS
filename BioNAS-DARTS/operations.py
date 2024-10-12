import torch
import torch.nn as nn
from biotorch.layers.utils import convert_layer
OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
 
  'skip_connect_u' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine, mode='usf'),
  'skip_connect_b' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine, mode='brsf'),
  'skip_connect_f' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine, mode="frsf"),

  'sep_conv_3x3_u' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine, mode='usf'),
  'sep_conv_3x3_br' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine, mode='brsf'),
  'sep_conv_3x3_fr' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine, mode='frsf'),
  'sep_conv_3x3_fa' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine, mode='fa'),

  'sep_conv_5x5_u' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine, mode='usf'),
  'sep_conv_5x5_br' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine, mode='brsf'),
  'sep_conv_5x5_fr' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine, mode='frsf'),
  'sep_conv_5x5_fa' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine, mode='fa'),

  'sep_conv_7x7_u' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine, mode='usf'),
  'sep_conv_7x7_br' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine, mode='brsf'),
  'sep_conv_7x7_fr' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine, mode='frsf'),
  'sep_conv_7x7_fa' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine, mode='fa'),


  'dil_conv_3x3_u' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine, mode='usf'),
  'dil_conv_3x3_br' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine, mode='brsf'),
  'dil_conv_3x3_fr' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine, mode='frsf'),
  'dil_conv_3x3_fa' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine, mode='fa'),


  'dil_conv_5x5_u' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine, mode='usf'),
  'dil_conv_5x5_br' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine, mode='brsf'),
  'dil_conv_5x5_fr' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine, mode='frsf'),
  'dil_conv_5x5_fa' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine, mode='fa'),

  # add mode
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
}

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)
# add mode
class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, mode='backpropagation'):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      convert_layer(nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False), mode=mode, copy_weights=False),
      convert_layer(nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), mode=mode, copy_weights=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)

# add mode
class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, mode='backpropagation'):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      convert_layer(nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False), mode=mode, copy_weights=False),
      convert_layer(nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False), mode=mode, copy_weights=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      convert_layer(nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False), mode=mode, copy_weights=False),
      convert_layer(nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), mode=mode, copy_weights=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)

# add mode
class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True, mode='backpropagation'):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    convert_layer(nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False), mode=mode, copy_weights=False),

    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    convert_layer(nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False), mode=mode, copy_weights=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out



