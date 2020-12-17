# This file was modified from https://github.com/BobLiu20/YOLOv3_PyTorch
# It needed to be modified in order to accomodate for different strides in the

import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

import torch
from torch import nn
from torch.nn import functional as F


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)



class NLBlockND(nn.Module):
  def __init__(self, in_channels, inter_channels=None, mode='embedded',
               dimension=3, bn_layer=True):
    """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
    args:
        in_channels: original channel size (1024 in the paper)
        inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
        mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
        dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
        bn_layer: whether to add batch norm
    """
    super(NLBlockND, self).__init__()

    assert dimension in [1, 2, 3]

    if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
      raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

    self.mode = mode
    self.dimension = dimension

    self.in_channels = in_channels
    self.inter_channels = inter_channels

    # the channel size is reduced to half inside the block
    if self.inter_channels is None:
      self.inter_channels = in_channels // 2
      if self.inter_channels == 0:
        self.inter_channels = 1

    # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
    if dimension == 3:
      conv_nd = nn.Conv3d
      max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
      bn = nn.BatchNorm3d
    elif dimension == 2:
      conv_nd = nn.Conv2d
      max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
      bn = nn.BatchNorm2d
    else:
      conv_nd = nn.Conv1d
      max_pool_layer = nn.MaxPool1d(kernel_size=(2))
      bn = nn.BatchNorm1d

    # function g in the paper which goes through conv. with kernel size 1
    self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

    # add BatchNorm layer after the last conv layer
    if bn_layer:
      self.W_z = nn.Sequential(
        conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
        bn(self.in_channels)
      )
      # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
      nn.init.constant_(self.W_z[1].weight, 0)
      nn.init.constant_(self.W_z[1].bias, 0)
    else:
      self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

      # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
      nn.init.constant_(self.W_z.weight, 0)
      nn.init.constant_(self.W_z.bias, 0)

    # define theta and phi for all operations except gaussian
    if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
      self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
      self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

    if self.mode == "concatenate":
      self.W_f = nn.Sequential(
        nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
        nn.ReLU()
      )

  def forward(self, x):
    """
    args
        x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
    """

    batch_size = x.size(0)

    # (N, C, THW)
    # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
    g_x = self.g(x).view(batch_size, self.inter_channels, -1)
    g_x = g_x.permute(0, 2, 1)

    if self.mode == "gaussian":
      theta_x = x.view(batch_size, self.in_channels, -1)
      phi_x = x.view(batch_size, self.in_channels, -1)
      theta_x = theta_x.permute(0, 2, 1)
      f = torch.matmul(theta_x, phi_x)

    elif self.mode == "embedded" or self.mode == "dot":
      theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
      phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
      theta_x = theta_x.permute(0, 2, 1)
      f = torch.matmul(theta_x, phi_x)

    elif self.mode == "concatenate":
      theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
      phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

      h = theta_x.size(2)
      w = phi_x.size(3)
      theta_x = theta_x.repeat(1, 1, 1, w)
      phi_x = phi_x.repeat(1, 1, h, 1)

      concat = torch.cat([theta_x, phi_x], dim=1)
      f = self.W_f(concat)
      f = f.view(f.size(0), f.size(2), f.size(3))

    if self.mode == "gaussian" or self.mode == "embedded":
      f_div_C = F.softmax(f, dim=-1)
    elif self.mode == "dot" or self.mode == "concatenate":
      N = f.size(-1)  # number of position in x
      f_div_C = f / N

    y = torch.matmul(f_div_C, g_x)

    # contiguous here just allocates contiguous chunk of memory
    y = y.permute(0, 2, 1).contiguous()
    y = y.view(batch_size, self.inter_channels, *x.size()[2:])

    W_y = self.W_z(y)
    # residual connection
    out = W_y + x

    return out


# if __name__ == '__main__':
#   import torch
#
#   for bn_layer in [True, False]:
#     img = torch.zeros(2, 3, 20)
#     net = NLBlockND(in_channels=3, mode='concatenate', dimension=1, bn_layer=bn_layer)
#     out = net(img)
#     print(out.size())
#
#     img = torch.zeros(2, 3, 20, 20)
#     net = NLBlockND(in_channels=3, mode='concatenate', dimension=2, bn_layer=bn_layer)
#     out = net(img)
#     print(out.size())
#
#     img = torch.randn(2, 3, 8, 20, 20)
#     net = NLBlockND(in_channels=3, mode='concatenate', dimension=3, bn_layer=bn_layer)
#     out = net(img)
#     print(out.size())

class BasicBlock(nn.Module):
  def __init__(self, inplanes, planes, bn_d=0.1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                           stride=1, padding=0, bias=False)
    self.bn1 = nn.BatchNorm2d(planes[0], momentum=bn_d)
    self.relu1 = nn.LeakyReLU(0.1)
    self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes[1], momentum=bn_d)
    self.relu2 = nn.LeakyReLU(0.1)

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu2(out)

    out += residual
    return out


# ******************************************************************************

# number of layers per model
model_blocks = {
    21: [1, 1, 2, 2, 1],
    53: [1, 2, 8, 8, 4],
}


class Backbone(nn.Module):
  """
     Class for DarknetSeg. Subclasses PyTorch's own "nn" module
  """

  def __init__(self, params,non_local=False):
    super(Backbone, self).__init__()
    self.use_range = params["input_depth"]["range"]
    self.use_xyz = params["input_depth"]["xyz"]
    self.use_remission = params["input_depth"]["remission"]
    self.drop_prob = params["dropout"]
    self.bn_d = params["bn_d"]
    self.OS = params["OS"]
    self.layers = params["extra"]["layers"]
    print("Using DarknetNet" + str(self.layers) + " Backbone")
    self.apply(_weights_init)

    # input depth calc
    self.input_depth = 0
    self.input_idxs = []
    if self.use_range:
      self.input_depth += 1
      self.input_idxs.append(0)
    if self.use_xyz:
      self.input_depth += 3
      self.input_idxs.extend([1, 2, 3])
    if self.use_remission:
      self.input_depth += 1
      self.input_idxs.append(4)
    print("Depth of backbone input = ", self.input_depth)

    # stride play
    self.strides = [2, 2, 2, 2, 2]
    # check current stride
    current_os = 1
    for s in self.strides:
      current_os *= s
    print("Original OS: ", current_os)

    # make the new stride
    if self.OS > current_os:
      print("Can't do OS, ", self.OS,
            " because it is bigger than original ", current_os)
    else:
      # redo strides according to needed stride
      for i, stride in enumerate(reversed(self.strides), 0):
        if int(current_os) != self.OS:
          if stride == 2:
            current_os /= 2
            self.strides[-1 - i] = 1
          if int(current_os) == self.OS:
            break
      print("New OS: ", int(current_os))
      print("Strides: ", self.strides)

    # check that darknet exists
    assert self.layers in model_blocks.keys()

    # generate layers depending on darknet type
    self.blocks = model_blocks[self.layers]

    # input layer
    self.conv1 = nn.Conv2d(self.input_depth, 32, kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(32, momentum=self.bn_d)
    self.relu1 = nn.LeakyReLU(0.1)

    # encoder
    self.enc1 = self._make_enc_layer(BasicBlock, [32, 64], self.blocks[0],
                                     stride=self.strides[0], bn_d=self.bn_d)
    self.enc2 = self._make_enc_layer(BasicBlock, [64, 128], self.blocks[1],
                                     stride=self.strides[1], bn_d=self.bn_d)
    self.enc3 = self._make_enc_layer(BasicBlock, [128, 256], self.blocks[2],
                                     stride=self.strides[2], bn_d=self.bn_d)
    self.enc4 = self._make_enc_layer(BasicBlock, [256, 512], self.blocks[3],
                                     stride=self.strides[3], bn_d=self.bn_d)
    self.enc5 = self._make_enc_layer(BasicBlock, [512, 1024], self.blocks[4],
                                     stride=self.strides[4], bn_d=self.bn_d,non_local=non_local)

    # for a bit of fun
    self.dropout = nn.Dropout2d(self.drop_prob)

    # last channels
    self.last_channels = 1024

  # make layer useful function
  def _make_enc_layer(self, block, planes, blocks, stride, bn_d=0.1,non_local=False):
    layers = []

    #  downsample
    layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                     kernel_size=3,
                                     stride=[1, stride], dilation=1,
                                     padding=1, bias=False)))
    layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
    layers.append(("relu", nn.LeakyReLU(0.1)))

    #  blocks
    inplanes = planes[1]
    if non_local:
      # for i in range(0,int(blocks/2)):
      for i in range(0,int(blocks-1)):
        # layers.append(("non_local_{}".format(i),NLBlockND(in_channels=inplanes, dimension=2)))
        layers.append(("residual_{}".format(i),
                       block(inplanes, planes, bn_d)))
      layers.append(("non_local_{}".format(0), NLBlockND(in_channels=inplanes, dimension=2)))
      layers.append(("residual_{}".format(i+1),
                   block(inplanes, planes, bn_d)))
    else:
      for i in range(0, blocks):
        layers.append(("residual_{}".format(i),
                       block(inplanes, planes, bn_d)))

    return nn.Sequential(OrderedDict(layers))

  def run_layer(self, x, layer, skips, os):
    y = layer(x)
    if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
      skips[os] = x.detach()
      os *= 2
    x = y
    return x, skips, os

  def forward(self, x):
    # filter input
    x = x[:, self.input_idxs]

    # run cnn
    # store for skip connections
    skips = {}
    os = 1

    # first layer
    x, skips, os = self.run_layer(x, self.conv1, skips, os)
    x, skips, os = self.run_layer(x, self.bn1, skips, os)
    x, skips, os = self.run_layer(x, self.relu1, skips, os)

    # all encoder blocks with intermediate dropouts
    x, skips, os = self.run_layer(x, self.enc1, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.enc2, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.enc3, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.enc4, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.enc5, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)

    return x, skips

  def get_last_depth(self):
    return self.last_channels

  def get_input_depth(self):
    return self.input_depth

def backbonenon(params,non_local=False, **kwargs):
    """Constructs a ResNet-56 model.
    """
    return Backbone(params,non_local=non_local, **kwargs)
