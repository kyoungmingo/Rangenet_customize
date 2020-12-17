# ******************************************************************************

# number of layers per model
# darknet 21인지 53인지 layer 수를 정의한다.
model_blocks = {
    21: [1, 1, 2, 2, 1],
    53: [1, 2, 8, 8, 4],
}

class Backbone(nn.Module):
  """
     Class for DarknetSeg. Subclasses PyTorch's own "nn" module
  """

  def __init__(self, params):
    super(Backbone, self).__init__()
    self.use_range = params["input_depth"]["range"]
    self.use_xyz = params["input_depth"]["xyz"]
    self.use_remission = params["input_depth"]["remission"]
    self.drop_prob = params["dropout"]
    self.bn_d = params["bn_d"]
    self.OS = params["OS"]
    self.layers = params["extra"]["layers"]
    print("Using DarknetNet" + str(self.layers) + " Backbone")

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
    # model/module을 구성하는 Layer를 생성
    self.conv1 = nn.Conv2d(self.input_depth, 32, kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(32, momentum=self.bn_d)
    self.relu1 = nn.LeakyReLU(0.1)

    # encoder
    # BasicBlock class는 encoder module을 편하게 생성하기 위함인가?
    self.enc1 = self._make_enc_layer(BasicBlock, [32, 64], self.blocks[0],
                                     stride=self.strides[0], bn_d=self.bn_d)
    self.enc2 = self._make_enc_layer(BasicBlock, [64, 128], self.blocks[1],
                                     stride=self.strides[1], bn_d=self.bn_d)
    self.enc3 = self._make_enc_layer(BasicBlock, [128, 256], self.blocks[2],
                                     stride=self.strides[2], bn_d=self.bn_d)
    self.enc4 = self._make_enc_layer(BasicBlock, [256, 512], self.blocks[3],
                                     stride=self.strides[3], bn_d=self.bn_d)
    self.enc5 = self._make_enc_layer(BasicBlock, [512, 1024], self.blocks[4],
                                     stride=self.strides[4], bn_d=self.bn_d)

    # for a bit of fun
    self.dropout = nn.Dropout2d(self.drop_prob)

    # last channels
    self.last_channels = 1024

  # make layer useful function
  def _make_enc_layer(self, block, planes, blocks, stride, bn_d=0.1):
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
    for i in range(0, blocks):
      layers.append(("residual_{}".format(i),
                     block(inplanes, planes, bn_d)))
    #nn.Sequential Layer를 쌓아 만든 여러개의 Module을 연속적으로 연결한다.
    #OrderedDict를 통해 순서를 기억한다?
    return nn.Sequential(OrderedDict(layers))

  def run_layer(self, x, layer, skips, os):
    y = layer(x)
    if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
      skips[os] = x.detach()
      os *= 2
    x = y
    return x, skips, os
  #Output을 정의한다.
  #x, skips, outstride(os)를 return한다.
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


# get the model
bboneModule = imp.load_source("bboneModule",
                              '/mnt/han/lidar-bonnetal/train/' + '/backbones/' +
                              ARCH["backbone"]["name"] + '.py')

backbone = bboneModule.Backbone(params=ARCH["backbone"])

#residual net
#skip도 존재
print(backbone)
#156개의 parameters 존재
print(len(list(backbone.parameters())))
#[0]이 첫번째 conv layer이고 requires_grad = True
#conv와 같은 module은 requires_grad = True가 default인 듯

# do a pass of the backbone to initialize the skip connections
# skip connection을 init하겠
stub = torch.zeros((1,
                    backbone.get_input_depth(),
                    ARCH["dataset"]["sensor"]["img_prop"]["height"],
                    ARCH["dataset"]["sensor"]["img_prop"]["width"]))

if torch.cuda.is_available():
  stub = stub.cuda()
  backbone.cuda()
_, stub_skips = backbone(stub)

decoderModule = imp.load_source("decoderModule",
                                    '/mnt/han/lidar-bonnetal/train' + '/tasks/semantic/decoders/' +
                                    ARCH["decoder"]["name"] + '.py')
decoder = decoderModule.Decoder(params=ARCH["decoder"],
                                     stub_skips=stub_skips,
                                     OS=ARCH["backbone"]["OS"],
                                     feature_depth=backbone.get_last_depth())

nclasses = len(learning_map_inv)

head = nn.Sequential(nn.Dropout2d(p=ARCH["head"]["dropout"]),
                          nn.Conv2d(decoder.get_last_depth(),
                                    nclasses, kernel_size=3,
                                    stride=1, padding=1))


#False라고 나옴
#if ARCH["post"]["CRF"]["use"]:
CRF = CRF(ARCH["post"]["CRF"]["train"], nclasses)
#else:
#  CRF = None

weights_enc = sum(p.numel() for p in backbone.parameters())
weights_dec = sum(p.numel() for p in decoder.parameters())
weights_head = sum(p.numel() for p in head.parameters())
print("Param encoder ", weights_enc)
print("Param decoder ", weights_dec)
print("Param head ", weights_head)
if CRF:
  weights_crf = sum(p.numel() for p in CRF.parameters())
  print("Param CRF ", weights_crf)

#  def forward(self, x, mask=None):
y, skips = backbone(stub)

#non-local block(batch_size,channel,H,W)
stub2 = stub.view(1,5,-1)
#to dot-product
stub3 = stub2.permute(0,2,1)


#.cuda() 맞춰야함
#기존 segmentator.py엔 .cuda를 따로 지정하지 않았는데 어떻게 돌아가지?
decoder.cuda()
y = decoder(y, skips)
head.cuda()
y = head(y)
y = F.softmax(y, dim=1)

if CRF:
  assert(mask is not None)
  y = CRF(x, y, mask)


# GPU 할당 변경하기
GPU_NUM = 0 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check

xx=torch.unsqueeze(first_batch[0][0],0)
xx=xx.cuda()
xxmask=torch.unsqueeze(first_batch[1][0],0)

backbone.cuda()
y, skips = backbone(xx)
decoder.cuda()
y = decoder(y, skips)
head.cuda()
y = head(y)
y = F.softmax(y, dim=1)

#proj_labels
#first_batch[2][0].shape
xxlabel=torch.unsqueeze(first_batch[2][0],0)

parserModule = imp.load_source("parserModule",
                               #    TRAIN_PATH
                               '/mnt/han/lidar-bonnetal/train' + '/tasks/semantic/dataset/' +
                                   DATA["name"] + '/parser.py')
datadir = FLAGS.dataset
logdir = FLAGS.log
parser = parserModule.Parser(root=datadir,
                                      train_sequences=DATA["split"]["train"],
                                      valid_sequences=DATA["split"]["valid"],
                                      test_sequences=None,
                                      labels=DATA["labels"],
                                      color_map=DATA["color_map"],
                                      learning_map=DATA["learning_map"],
                                      learning_map_inv=DATA["learning_map_inv"],
                                      sensor=ARCH["dataset"]["sensor"],
                                      max_points=ARCH["dataset"]["max_points"],
                                      batch_size=ARCH["train"]["batch_size"],
                                      workers=ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=True)

epsilon_w = ARCH["train"]["epsilon_w"]
content = torch.zeros(parser.get_n_classes(), dtype=torch.float)
for cl, freq in DATA["content"].items():
  x_cl = parser.to_xentropy(cl)  # map actual class to xentropy class
  content[x_cl] += freq
loss_w = 1 / (content + epsilon_w)   # get weights
for x_cl, w in enumerate(loss_w):  # ignore the ones necessary to ignore
  if DATA["learning_ignore"][x_cl]:
    # don't weigh
    loss_w[x_cl] = 0
print("Loss weights from content: ", loss_w.data)

#해당 과정을 진행 후에, loss 계산이 됐다.
#output 20,64,2048
#labels 64,2048

xxlabel = xxlabel.cuda(non_blocking=True).long()

criterion = nn.NLLLoss(weight=loss_w).to(device)
criterion = nn.DataParallel(criterion).cuda()

loss = criterion(torch.log(y.clamp(min=1e-8)), xxlabel)
loss.backward()

#argmax.shape 64,2048
# measure accuracy and record loss
argmax = y.argmax(dim=1)
#evaluator.addBatch(argmax, xxlabel)
#losses.update(loss.mean().item(), in_vol.size(0))

# optimizer
if self.ARCH["post"]["CRF"]["use"] and self.ARCH["post"]["CRF"]["train"]:
    self.lr_group_names = ["post_lr"]
    self.train_dicts = [{'params': self.model_single.CRF.parameters()}]
else:
    self.lr_group_names = []
    self.train_dicts = []
if self.ARCH["backbone"]["train"]:
    self.lr_group_names.append("backbone_lr")
    self.train_dicts.append(
        {'params': self.model_single.backbone.parameters()})
if self.ARCH["decoder"]["train"]:
    self.lr_group_names.append("decoder_lr")
    self.train_dicts.append(
        {'params': self.model_single.decoder.parameters()})
if self.ARCH["head"]["train"]:
    self.lr_group_names.append("head_lr")
    self.train_dicts.append({'params': self.model_single.head.parameters()})

# Use SGD optimizer to train
self.optimizer = optim.SGD(self.train_dicts,
                           lr=self.ARCH["train"]["lr"],
                           momentum=self.ARCH["train"]["momentum"],
                           weight_decay=self.ARCH["train"]["w_decay"])

# Use warmup learning rate
# post decay and step sizes come in epochs and we want it in steps
steps_per_epoch = self.parser.get_train_size()
up_steps = int(self.ARCH["train"]["wup_epochs"] * steps_per_epoch)
final_decay = self.ARCH["train"]["lr_decay"] ** (1 / steps_per_epoch)
self.scheduler = warmupLR(optimizer=self.optimizer,
                          lr=self.ARCH["train"]["lr"],
                          warmup_steps=up_steps,
                          momentum=self.ARCH["train"]["momentum"],
                          decay=final_decay)


#####################200720 prediction shape 확인#########################

#데이터 불러오기

import numpy as np

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

root='/mnt/han/lidar-bonnetal/train/tasks/semantic/runs/predictions_1'

scan_path = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/dataset/dataset/", "sequences","00", "velodyne")
class_path = os.path.join(root,  "sequences","00", "predictions")
label_path = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/dataset/dataset/", "sequences","00", "labels")

scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(class_path)) for f in fn if is_label(f)]
# label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
#           os.path.expanduser(label_path)) for f in fn if is_label(f)]


scan_files.sort()
label_files.sort()

scan_file = scan_files[0]
label_file = label_files[0]

scan = np.fromfile(scan_file, dtype=np.float32)
scan = scan.reshape((-1, 4))

# put in attribute
points = scan[:, 0:3]  # get xyz
remissions = scan[:, 3]  # get remission

# if all goes well, open label
label = np.fromfile(label_file, dtype=np.int32)
label = label.reshape((-1))

#label 위치 찾기(output)
np.where(label==10)[0]

#label이 10인 point 0으로 전환
scan[np.where(label==10)] = 0

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger

from tasks.semantic.modules.trainer import *


import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import imp
import yaml
import time
from PIL import Image
import __init__ as booger
import collections
import copy
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

from common.logger import Logger
from common.avgmeter import *
from common.sync_batchnorm.batchnorm import convert_model
from common.warmupLR import *
from tasks.semantic.modules.segmentator import *
from tasks.semantic.modules.ioueval import *

parser = argparse.ArgumentParser("./train.py")
parser.add_argument(
      '--dataset', '-d',
      type=str,
      #required=True,
      required=False,
      default='/mnt/han/lidar-bonnetal/train/tasks/semantic/dataset/dataset',
      help='Dataset to train with. No Default',
  )
parser.add_argument(
      '--arch_cfg', '-ac',
      type=str,
      #required=True,
      required=False,
      default='/mnt/han/lidar-bonnetal/train/tasks/semantic/config/arch/darknet53.yaml',
      help='Architecture yaml cfg file. See /config/arch for sample. No default!',
  )
parser.add_argument(
      '--data_cfg', '-dc',
      type=str,
      required=False,
      default='/mnt/han/lidar-bonnetal/train/tasks/semantic/config/labels/data_cfg.yaml',
      help='Classification yaml cfg file. See /config/labels for sample. No default!',
  )
parser.add_argument(
      '--log', '-l',
      type=str,
      required=False,
      default='/mnt/han/lidar-bonnetal/train/tasks/semantic/log_2',
      #default=os.path.expanduser("~") + '/logs/' +
      #datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
      help='Directory to put the log data. Default: ~/logs/date+time'
  )
parser.add_argument(
      '--pretrained', '-p',
      type=str,
      required=False,
      default=None,
      help='Directory to get the pretrained model. If not passed, do from scratch!'
  )
FLAGS, unparsed = parser.parse_known_args()

#####parser를 통해서 path 및 정보 가져오기#####


#TRAIN_PATH 설정
import sys
TRAIN_PATH = "../../"
DEPLOY_PATH = "../../../deploy"
sys.path.insert(0, TRAIN_PATH)

ARCH = yaml.safe_load(open(FLAGS.arch_cfg, 'r'))
DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))






#LaserScan,SemLaserScan 함수 필요 >> 함수 실행 >> parser.py에서 함수 주석처리
parserModule = imp.load_source("parserModule",
                               #    TRAIN_PATH
                               '/mnt/han/lidar-bonnetal/train' + '/tasks/semantic/dataset/' +
                                   DATA["name"] + '/parser_ENet.py')


datadir = FLAGS.dataset
logdir = FLAGS.log

#parser.py 모듈 가져오기

parser = parserModule.Parser(root=datadir,
                                      train_sequences=DATA["split"]["train"],
                                      valid_sequences=DATA["split"]["valid"],
                                      test_sequences=None,
                                      labels=DATA["labels"],
                                      color_map=DATA["color_map"],
                                      learning_map=DATA["learning_map"],
                                      learning_map_inv=DATA["learning_map_inv"],
                                      sensor=ARCH["dataset"]["sensor"],
                                      max_points=ARCH["dataset"]["max_points"],
                                      batch_size=ARCH["train"]["batch_size"],
                                      workers=ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=True)

# weights for loss (and bias)
# Freq, ex) roads 등의 target 불균형을 맞춰주기 위해 weight + term

epsilon_w = ARCH["train"]["epsilon_w"]

#class 20

content = torch.zeros(parser.get_n_classes(), dtype=torch.float)

for cl, freq in DATA["content"].items():
    x_cl = parser.to_xentropy(cl)  # map actual class to xentropy class
    content[x_cl] += freq
loss_w = 1 / (content + epsilon_w)   # get weights

# ignore the ones necessary to ignore
for x_cl, w in enumerate(loss_w):
    if DATA["learning_ignore"][x_cl]:
        # don't weigh
       loss_w[x_cl] = 0
print("Loss weights from content: ", loss_w.data)

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from common.laserscan_ENet import LaserScan, SemLaserScan

root = os.path.join(parser.root, "sequences")
sequences = [0]
labels = parser.labels
color_map = parser.color_map
learning_map = parser.learning_map
learning_map_inv = parser.learning_map_inv
sensor = parser.sensor
sensor_img_H = sensor["img_prop"]["height"]
sensor_img_W = sensor["img_prop"]["width"]
sensor_img_means = torch.tensor(sensor["img_means"],
                                     dtype=torch.float)
sensor_img_stds = torch.tensor(sensor["img_stds"],
                                    dtype=torch.float)
sensor_fov_up = sensor["fov_up"]
sensor_fov_down = sensor["fov_down"]
max_points = parser.max_points
gt = parser.gt

nclasses = len(parser.learning_map_inv)


scan = SemLaserScan(color_map,
                          project=True,
                          H=sensor_img_H,
                          W=sensor_img_W,
                          fov_up=sensor_fov_up,
                          fov_down=sensor_fov_down)
scan_files = []
label_files = []
class_files = []

# fill in with names, checking that all sequences are complete
# for seq in sequences:
#     # to string
#     seq = '{0:02d}'.format(int(seq))
#
#     print("parsing seq {}".format(seq))

    # get paths for each
scan_path = os.path.join(root, "01", "velodyne")
label_path = os.path.join(root, "01", "labels")
class_path = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/DB_1/predcitions_stage1/sequences/", "01",
                          "predictions")

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

    # get files
scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(label_path)) for f in fn if is_label(f)]
class_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(class_path)) for f in fn if is_label(f)]

    # check all scans have labels
    # if gt:
assert (len(scan_files) == len(label_files) == len(class_files))

    # extend list
scan_files.extend(scan_files)
label_files.extend(label_files)
class_files.extend(class_files)

# sort for correspondance
scan_files.sort()
label_files.sort()
class_files.sort()
#
# print("Using {} scans from sequences {}".format(len(scan_files),
#                                                 sequences))


# def __getitem__(self, index):
    # get item in tensor shape
scan_file = scan_files[0]
class_file = class_files[0]
label_file = label_files[0]

gt = True

# open a semantic laserscan
if gt:
    scan = SemLaserScan(color_map,
                        project=True,
                        H=sensor_img_H,
                        W=sensor_img_W,
                        fov_up=sensor_fov_up,
                        fov_down=sensor_fov_down)
else:
    scan = LaserScan(project=True,
                     H=sensor_img_H,
                     W=sensor_img_W,
                     fov_up=sensor_fov_up,
                     fov_down=sensor_fov_down)

# open and obtain scan
scan.open_scan(scan_file,class_file)
if gt:
    scan.open_label(label_file)
    # map unused classes to used classes (also for projection)
    scan.sem_label = map(scan.sem_label, learning_map)
    scan.proj_sem_label = map(scan.proj_sem_label, learning_map)


#여기서 빼면 안될듯,,, projection이 안되는 현상 발생
#0을주고 0인 값을 없애면된다. 그냥 그 포인트 자체를 빼주면 된다..


scan = np.fromfile(scan_file, dtype=np.float32)
scan = scan.reshape((-1, 4))
eclass = np.fromfile(class_file, dtype=np.int32)
eclass = eclass.reshape((-1))
label = np.fromfile(label_file, dtype=np.int32)
label = label.reshape((-1))
# scan[np.where(eclass == 10)] = 9999
# scan[np.where(eclass == 40)] = 0
# scan[np.where(eclass == 70)] = 0



#해당 class 포인트 삭제
del_class = [10,40,48,50,70]

#scan input data point delete
for p in del_class:
    for i in np.where(eclass == p)[0]:
        scan[i] = 1000000000
scan = np.delete(scan,np.unique(np.where(scan == 1000000000)[0]),axis=0)

#label target data label delete
for p in del_class:
    for i in np.where(eclass == p)[0]:
        label[i] = 1000000000
label = np.delete(label,np.unique(np.where(label == 1000000000)[0]))

for p in self.del_class:
    for i in np.where(self.eclass == p)[0]:
        label[i] = 1000000000
label = np.delete(label, np.unique(np.where(label == 1000000000)[0]))











EXTENSIONS_LABEL = ['.label']

def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


class_path1 = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/runs_4/predictions_ENet",  "sequences","08", "predictions")
class_path2 = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/runs_stage2/predictions_stage2",  "sequences","08", "predictions")


   # get files
class_files1 = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(class_path1)) for f in fn if is_label(f)]
class_files2 = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(class_path2)) for f in fn if is_label(f)]

    # check all scans have labels
    # if gt:
assert (len(class_files1) == len(class_files2))

     # extend list
# scan_files.extend(scan_files)
# label_files.extend(label_files)
# class_files.extend(class_files)

# sort for correspondance
class_files1.sort()
class_files2.sort()

os.makedirs(os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/final_pred", "sequences",
                    "08"))

for i in np.arange(len(class_files1)):
    class_file1 = class_files1[i]
    class_file2 = class_files2[i]

    path_norm = os.path.normpath(class_file1)
    path_split = path_norm.split(os.sep)
    path_name = path_split[-1].replace(".bin", ".label")


    eclass1 = np.fromfile(class_file1, dtype=np.int32)
    eclass1 = eclass1.reshape((-1))

    eclass2 = np.fromfile(class_file2, dtype=np.int32)
    eclass2 = eclass2.reshape((-1))

    eclass1[np.where(eclass1 == 11)] = eclass2

    path = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/final_pred", "sequences",
                    "08", path_name)
    eclass1.tofile(path)








scan_names = []
# sequnece = '08'
sequence = '{0:02d}'.format(int(sequence))
scan_paths = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/dataset/dataset/", "sequences",
                          str(sequence), "velodyne")
# populate the scan names
seq_scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(scan_paths)) for f in fn if ".bin" in f]
seq_scan_names.sort()
scan_names.extend(seq_scan_names)
# print(scan_names)

# get label paths
label_names = []
class_names = []

sequence = '08'
sequence = '{0:02d}'.format(int(sequence))
label_paths = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/dataset/dataset/", "sequences",
                           str(sequence), "labels")
class_paths = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/runs_4/predictions_ENet/sequences/", str(sequence),
                          "predictions")

# populate the label names
seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(label_paths)) for f in fn if ".label" in f]
seq_label_names.sort()
label_names.extend(seq_label_names)

seq_class_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(class_paths)) for f in fn if ".label" in f]
seq_class_names.sort()
class_names.extend(seq_class_names)
# print(label_names)

# get predictions paths
pred_names = []

sequence = '08'
sequence = '{0:02d}'.format(int(sequence))
pred_paths = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/runs_stage2/predictions_stage2/", "sequences",
                          sequence, "predictions")
# populate the label names
seq_pred_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(pred_paths)) for f in fn if ".label" in f]
seq_pred_names.sort()
pred_names.extend(seq_pred_names)
# print(pred_names)

# check that I have the same number of files
# print("labels: ", len(label_names))
# print("predictions: ", len(pred_names))
assert(len(label_names) == len(scan_names) and
     len(label_names) == len(pred_names))



#여기서 빼면 안될듯,,, projection이 안되는 현상 발생
#0을주고 0인 값을 없애면된다. 그냥 그 포인트 자체를 빼주면 된다..
scan_file=scan_names[0]
label_file=label_names[0]
class_file=class_names[0]
pred_file=pred_names[0]


scan = np.fromfile(scan_file, dtype=np.float32)
scan = scan.reshape((-1, 4))
eclass = np.fromfile(class_file, dtype=np.int32)
eclass = eclass.reshape((-1))
label = np.fromfile(label_file, dtype=np.int32)
label = label.reshape((-1))
# scan[np.where(eclass == 10)] = 9999
# scan[np.where(eclass == 40)] = 0
# scan[np.where(eclass == 70)] = 0
pred = np.fromfile(pred_file, dtype=np.int32)
pred = pred.reshape((-1))


#해당 class 포인트 삭제
del_class = [10,40,48,50,70]

#scan input data point delete
for p in del_class:
    for i in np.where(eclass == p)[0]:
        scan[i] = 1000000000
scan = np.delete(scan,np.unique(np.where(scan == 1000000000)[0]),axis=0)

#label target data label delete
for p in del_class:
    for i in np.where(eclass == p)[0]:
        label[i] = 1000000000
label = np.delete(label,np.unique(np.where(label == 1000000000)[0]))

for p in self.del_class:
    for i in np.where(self.eclass == p)[0]:
        label[i] = 1000000000
label = np.delete(label, np.unique(np.where(label == 1000000000)[0]))

import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tasks.semantic.postproc.CRF import CRF
import __init__ as booger

bboneModule = imp.load_source("bboneModule",'/mnt/han/lidar-bonnetal/train/backbones/darknet_attention.py')
backbone1 = bboneModule.Backbone(params=ARCH["backbone"])
backbone = bboneModule.backbonenon(params=ARCH["backbone"],non_local="TRUE")

w_dict = torch.load("/mnt/han/lidar-bonnetal/train/tasks/semantic/DB_2/backbone",
                            map_location=lambda storage, loc: storage)
backbone.load_state_dict(w_dict, strict=False)
backbone1.load_state_dict(w_dict, strict=True)

print("Successfully loaded model backbone weights")

########competition 제출용 결과 만들기#####0825#############

import numpy as np
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from common.laserscan_ENet import LaserScan, SemLaserScan

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']

def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

stage1_path = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/darknet_pre/darknet53/sequences/00/predictions")
stage2_path = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/darknet_pre/pretrained/predictions_TT/sequences/00/predictions")

stage1_files = []
stage2_files = []

stage1_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(stage1_path)) for f in fn if is_label(f)]
stage2_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(stage2_path)) for f in fn if is_label(f)]

len(stage1_files) == len(stage2_files)

# sort for correspondance
stage1_files.sort()
stage2_files.sort()

#stage1의 points 불러오기
stage1 = np.fromfile(stage1_files[0], dtype=np.int32)
stage1 = stage1.reshape((-1))

#stage2의 points 불러오기
stage2 = np.fromfile(stage2_files[0], dtype=np.int32)
stage2 = stage2.reshape((-1))

#stage1의 class=11을 stage2의 class로 채워주기
k = 0

for i in np.unique(np.where(stage1 == 11)[0]):
    stage1[i] = stage2[k]
    k += 1

#검증하기
stage = np.fromfile(stage1_files[0], dtype=np.int32)
stage = stage.reshape((-1))

k = 0

for i in np.unique(np.where(stage == 11)[0]):
    assert(stage1[i] == stage2[k])
    k += 1


##전체 predictions들 값 대체하기

EXTENSIONS_LABEL = ['.label']

# def is_scan(filename):
#   return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

DATA = yaml.safe_load(open('/mnt/han/lidar-bonnetal/train/tasks/semantic/config/labels/data_cfg.yaml', 'r'))

test_sequences=DATA["split"]["valid"]

for seq in test_sequences:
    # to string
    seq = '{0:02d}'.format(int(seq))

    print("parsing seq {}".format(seq))

    os.makedirs(os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/finalkitti", "sequences", seq))
    os.makedirs(os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/finalkitti", "sequences", seq, "predictions"))

    stage1_path = os.path.join(
        "/mnt/han/lidar-bonnetal/train/tasks/semantic/DB_1/predcitions_stage1/sequences/",seq,"predictions")
    stage2_path = os.path.join(
        "/mnt/han/lidar-bonnetal/train/tasks/semantic/DB_2/predictions_stage2/sequences/",seq,"predictions")

    stage1_files = []
    stage2_files = []

    stage1_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(stage1_path)) for f in fn if is_label(f)]
    stage2_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(stage2_path)) for f in fn if is_label(f)]

    assert(len(stage1_files) == len(stage2_files))

    # sort for correspondance
    stage1_files.sort()
    stage2_files.sort()

    for d in np.arange(len(stage1_files)):
        stage1_file = stage1_files[d]
        stage2_file = stage2_files[d]
        # stage1의 points 불러오기
        stage1 = np.fromfile(stage1_file, dtype=np.int32)
        stage1 = stage1.reshape((-1))

        # stage2의 points 불러오기
        stage2 = np.fromfile(stage2_file, dtype=np.int32)
        stage2 = stage2.reshape((-1))

        # stage1의 class=11을 stage2의 class로 채워주기
        k = 0

        for i in np.unique(np.where(stage1 == 11)[0]):
            stage1[i] = stage2[k]
            k += 1

        # 검증하기
        stage = np.fromfile(stage1_file, dtype=np.int32)
        stage = stage.reshape((-1))

        k = 0

        for i in np.unique(np.where(stage == 11)[0]):
            assert (stage1[i] == stage2[k])
            k += 1

        #save pred
        path_norm = os.path.normpath(stage1_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")

        # save final
        path = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/finalkitti", "sequences",
                            path_seq, "predictions", path_name)
        stage1.tofile(path)



#############fusion DATA 준비#############

import numpy as np
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from common.laserscan_ENet import LaserScan, SemLaserScan

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']
EXTENSIONS_IMAGE = ['.png']


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

def is_image(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_IMAGE)


scan_files = []
label_files = []
image_files = []


for seq in train_sequences:
    # to string
    seq = '{0:02d}'.format(int(seq))

    print("parsing seq {}".format(seq))

    # get paths for each
    scan_path = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/cdataset/dataset/sequences", seq, "velodyne")
    label_path = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/cdataset/dataset/sequences", seq, "labels")
    image_path = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/cdataset/dataset/sequences", seq, "images")
    # get files
    scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
    label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
      os.path.expanduser(label_path)) for f in fn if is_label(f)]
    image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(image_path)) for f in fn if is_image(f)]


    # check all scans have labels
    # assert(len(scan_files) == len(label_files))
    k = 0

    fscan_files = []
    flabel_files = []
    fimage_files = []

    for i in np.arange(len(scan_files)):
        for j in np.arange(len(image_files)):
            if scan_files[i].split("/")[-1][:-4] == image_files[j].split("_")[-1][:-7]:
                fscan_files.append(scan_files[i])
                flabel_files.append(label_files[i])
                fimage_files.append(image_files[j])
                k += 1
    # front_241154029000.png
    # 241154029.bin
    # extend list
    fscan_files.extend(fscan_files)
    flabel_files.extend(flabel_files)
    fimage_files.extend(fimage_files)
    assert (len(fscan_files) == len(flabel_files) == len(fimage_files))



    # sort for correspondance
scan_files.sort()
label_files.sort()

print("Using {} scans from sequences {}".format(len(scan_files),
                                                sequences))















scan_path = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/carla/points/00/velodyne")
label_path = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/carla/points/00/labels")
image_path = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/cdataset/dataset/sequences", "00", "images")


scan_files = []
label_files = []
image_files = []


scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(label_path)) for f in fn if is_label(f)]
image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(image_path)) for f in fn if is_image(f)]



len(scan_files) == len(label_files) == len(image_files)

# fimage_files = image_files[0].split("/")[-1]

# sort for correspondance
scan_files.sort()
label_files.sort()
image_files.sort()

k=0

fscan_files = []
flabel_files = []
fimage_files = []

for i in np.arange(len(scan_files)):
    for j in np.arange(len(image_files)):
        if scan_files[i].split("/")[-1][:-4] == image_files[j].split("_")[-1][:10]:
                fscan_files.append(scan_files[i])
                flabel_files.append(label_files[i])
                fimage_files.append(image_files[j])
                k += 1

#############data 정리 끝!###################

#remission값이 마지막 채널에 0으로 들어가 있는 듯하다.#
scan1 = np.fromfile(scan_files[0], dtype=np.float32)
scan1 = scan1.reshape((-1, 4))
# scan1[:,3]

label1 = np.fromfile(label_files[0], dtype=np.int32)
label1 = label1.reshape((-1))

print(scan1.shape,label1.shape)

print(scan_files[0],label_files[0])


############################################

import yaml

ARCH = yaml.safe_load(open("/mnt/han/lidar-bonnetal/train/tasks/semantic/carla_pretrained/arch_cfg.yaml", 'r'))
DATA = yaml.safe_load(open("/mnt/han/lidar-bonnetal/train/tasks/semantic/carla_pretrained/data_cfg.yaml", 'r'))

# ARCH = yaml.safe_load(open("/mnt/han/lidar-bonnetal/train/tasks/semantic/config/arch/darknet53.yaml", 'r'))
# DATA = yaml.safe_load(open("/mnt/han/lidar-bonnetal/train/tasks/semantic/config/labels/semantic-kitti.yaml", 'r'))

# root=datadir
train_sequences=DATA["split"]["train"]
valid_sequences=DATA["split"]["valid"]
test_sequences=None
labels=DATA["labels"]
color_map=DATA["color_map"]
learning_map=DATA["learning_map"]
learning_map_inv=DATA["learning_map_inv"]
sensor=ARCH["dataset"]["sensor"]
max_points=ARCH["dataset"]["max_points"]
batch_size=ARCH["train"]["batch_size"]
workers=ARCH["train"]["workers"]
gt=True
shuffle_train=True

# self.root = os.path.join(root, "sequences")
# self.sequences = sequences
labels = labels
color_map = color_map
learning_map = learning_map
learning_map_inv = learning_map_inv
sensor = sensor
sensor_img_H = sensor["img_prop"]["height"]
sensor_img_W = sensor["img_prop"]["width"]
sensor_img_means = torch.tensor(sensor["img_means"],
                                     dtype=torch.float)
sensor_img_stds = torch.tensor(sensor["img_stds"],
                                    dtype=torch.float)
sensor_fov_up = sensor["fov_up"]
sensor_fov_down = sensor["fov_down"]
max_points = max_points
gt = gt

scan_file = fscan_files[0]
label_file = flabel_files[0]
image_file = fimage_files[0]


scan = SemLaserScan(color_map,
                    project=True,
                    H=sensor_img_H,
                    W=sensor_img_W,
                    fov_up=sensor_fov_up,
                    fov_down=sensor_fov_down)


scan.open_scan(scan_file)
# if self.gt:

#여기서 오류 1번째 발생,
#points size와 label size가 다름!!!

scan.open_label(label_file)
# map unused classes to used classes (also for projection)
scan.sem_label = map(scan.sem_label, learning_map)
scan.proj_sem_label = map(scan.proj_sem_label, learning_map)

#input image
import cv2
img = cv2.imread(image_file,cv2.IMREAD_COLOR)
b, g, r = cv2.split(img)  # img파일을 b,g,r로 분리
img2 = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

import skimage

# img2 = skimage.transform.resize(img2, (300, 512))
img3 = img2.transpose((2, 0, 1))
img4 = torch.from_numpy(img3).float()
transforms.ToTensor(img3)

_mean = img2.mean(axis=(0, 1)) / 255
_std = img2.std(axis=(0, 1)) / 255

print('shape   :', img2.shape)
print('RGB mean:', _mean)
print('RGB std :', _std)

aug_f = transforms.Compose([transforms.ToTensor()])
img = aug_f(img2)

print('augmented img shape:', img.shape)
print('augmented img mean :', img.mean(axis=(1, 2)))
print('augmented img std  :', img.std(axis=(1, 2)))



# make a tensor of the uncompressed data (with the max num points)
#input data 저장 공간을 만들어 놓는 것 같음.

unproj_n_points = scan.points.shape[0]
unproj_xyz = torch.full((max_points, 3), -1.0, dtype=torch.float)
unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
unproj_range = torch.full([max_points], -1.0, dtype=torch.float)
unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
unproj_remissions = torch.full([max_points], -1.0, dtype=torch.float)
unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)

#matching이 안돼서 오류남
if gt:
    unproj_labels = torch.full([max_points], -1.0, dtype=torch.int32)
    unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
else:
    unproj_labels = []

# get points and labels
proj_range = torch.from_numpy(scan.proj_range).clone()
proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
proj_remission = torch.from_numpy(scan.proj_remission).clone()
proj_mask = torch.from_numpy(scan.proj_mask)
if gt:
  proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
  proj_labels = proj_labels * proj_mask
else:
  proj_labels = []
proj_x = torch.full([max_points], -1, dtype=torch.long)
proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
proj_y = torch.full([max_points], -1, dtype=torch.long)
proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
proj = torch.cat([proj_range.unsqueeze(0).clone(),
                  proj_xyz.clone().permute(2, 0, 1),
                  proj_remission.unsqueeze(0).clone()])

#size 맞춰주고 nomalize해주는 과정#
proj = (proj - sensor_img_means[:, None, None]
        ) / sensor_img_stds[:, None, None]

proj = proj * proj_mask.float()

proj2 = torch.cat([proj,img4])




#########img stacking 코딩 완료!#######################
# get name and sequence
path_norm = os.path.normpath(scan_file)
path_split = path_norm.split(os.sep)
path_seq = path_split[-3]
path_name = path_split[-1].replace(".bin", ".label")

label1 = np.fromfile(label_files[0], dtype=np.int32)
label1 = label1.reshape((-1))

np.unique(np.where(label1 == 7)[0])
#########transfer learning############################
####backbone 1st layer channel stacking###############

import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tasks.semantic.postproc.CRF import CRF
import __init__ as booger


import yaml

# ARCH = yaml.safe_load(open("/mnt/han/lidar-bonnetal/train/tasks/semantic/carla_pretrained/arch_cfg.yaml", 'r'))
ARCH = yaml.safe_load(open("/mnt/han/lidar-bonnetal/train/tasks/semantic/arch/darknet53.yaml", 'r'))

# DATA = yaml.safe_load(open("/mnt/han/lidar-bonnetal/train/tasks/semantic/carla_pretrained/data_cfg.yaml", 'r'))

# get the model
bboneModule = imp.load_source("bboneModule",
                              "/mnt/han/lidar-bonnetal/train" + '/backbones/' +
                              ARCH["backbone"]["name"] + '.py')
backbone = bboneModule.Backbone(params=ARCH["backbone"])

w_dict = torch.load("/mnt/han/lidar-bonnetal/train/tasks/semantic/carla_pretrained" + "/backbone",
                            map_location=lambda storage, loc: storage)
wrange=w_dict['conv1.weight'][:,0,:,:].unsqueeze(1)

w_dict['conv1.weight'].shape

rgb_weight = torch.cat((wrange,wrange,wrange),dim=1)

w_dict['conv1.weight']=torch.cat((w_dict['conv1.weight'],rgb_weight),dim=1)

backbone.load_state_dict(w_dict, strict=True)

# w_dict2 = torch.load("/mnt/han/lidar-bonnetal/train/tasks/semantic/DB_1" + "/backbone",
#                             map_location=lambda storage, loc: storage)

if not ARCH["backbone"]["train"]:
    for w in backbone.parameters():
        w.requires_grad = False



###############1012 projection visualize to compare RGB IMAGE#####################

fscan_files = []
flabel_files = []
fimage_files = []

scan_path = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/cdataset/dataset/sequences", "00",
                               "velodyne")
label_path = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/cdataset/dataset/sequences", "00",
                        "labels")
image_path = os.path.join("/mnt/han/lidar-bonnetal/train/tasks/semantic/cdataset/dataset/sequences", "00",
                        "images")

scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
  os.path.expanduser(label_path)) for f in fn if is_label(f)]
image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
  os.path.expanduser(image_path)) for f in fn if is_image(f)]


scan_files.sort()
label_files.sort()
image_files.sort()
fscan_files = []
flabel_files = []
fimage_files = []

for i in np.arange(len(scan_files)):
    for j in np.arange(len(image_files)):
        if scan_files[i].split("/")[-1][:-4] == image_files[j].split("_")[-1][:-7]:
            fscan_files.append(scan_files[i])
            flabel_files.append(label_files[i])
            fimage_files.append(image_files[j])


fscan_files.sort()
flabel_files.sort()
fimage_files.sort()

scan_file = fscan_files[100]
image_file = fimage_files[100]
if gt:
  label_file = flabel_files[100]

# open a semantic laserscan
if gt:
  scan = SemLaserScan(color_map,
                      project=True,
                      H=sensor_img_H,
                      W=sensor_img_W,
                      fov_up=sensor_fov_up,
                      fov_down=sensor_fov_down)
else:
  scan = LaserScan(project=True,
                   H=sensor_img_H,
                   W=sensor_img_W,
                   fov_up=sensor_fov_up,
                   fov_down=sensor_fov_down)

# open and obtain scan
scan.open_scan(scan_file)
if gt:
    scan.open_label(label_file)
  # map unused classes to used classes (also for projection)
    #map함수는 Semantic kitti 내장 map 함수를 사용해야함. 아니면, python 기본 내장 map함수로 다음 진행이 이뤄지지 않음.
    scan.sem_label = map(scan.sem_label, learning_map)
    scan.proj_sem_label = map(scan.proj_sem_label, learning_map)

#input image
img = cv2.imread(image_file,cv2.IMREAD_COLOR)
b, g, r = cv2.split(img)  # img파일을 b,g,r로 분리
img2 = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge
# img2 = transform.resize(img2, (300, 512))
#resize할 필요 없음 matching되니까
# img3 = img2.transpose((2, 0, 1))
# img4 = torch.from_numpy(img3).float()

#img2[RGB] > normalize
img4 = aug_f(img2)

# make a tensor of the uncompressed data (with the max num points)
unproj_n_points = scan.points.shape[0]
unproj_xyz = torch.full((max_points, 3), -1.0, dtype=torch.float)
unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
unproj_range = torch.full([max_points], -1.0, dtype=torch.float)
unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
unproj_remissions = torch.full([max_points], -1.0, dtype=torch.float)
unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
if gt:
  unproj_labels = torch.full([max_points], -1.0, dtype=torch.int32)
  unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
else:
  unproj_labels = []

# get points and labels
proj_range = torch.from_numpy(scan.proj_range).clone()
proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
proj_remission = torch.from_numpy(scan.proj_remission).clone()
proj_mask = torch.from_numpy(scan.proj_mask)
if gt:
  proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
  proj_labels = proj_labels * proj_mask
else:
  proj_labels = []
proj_x = torch.full([max_points], -1, dtype=torch.long)
proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
proj_y = torch.full([max_points], -1, dtype=torch.long)
proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
proj = torch.cat([proj_range.unsqueeze(0).clone(),
                  proj_xyz.clone().permute(2, 0, 1),
                  proj_remission.unsqueeze(0).clone()])
proj = (proj - sensor_img_means[:, None, None]
        ) / sensor_img_stds[:, None, None]
proj = proj * proj_mask.float()
proj = torch.cat([proj, img4])

# get name and sequence
path_norm = os.path.normpath(scan_file)
path_split = path_norm.split(os.sep)
path_seq = path_split[-3]
path_name = path_split[-1].replace(".bin", ".label")
# print("path_norm: ", path_norm)
# print("path_seq", path_seq)
# print("path_name", path_name)


label = map(proj_labels, learning_map_inv)
final = map(label, color_map)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.imshow(final)
plt.show()

root = '/mnt/han/lidar-bonnetal/train/tasks/semantic/cdataset/visualize/'

new_img_path = os.path.dirname(root)
new_point_path = os.path.dirname(root)
new_point_path += r'/pointcloud3.jpg' # jpg가 아닌 png 등도 가능하니다.
new_img_path += r'/RGB4.jpg' # jpg가 아닌 png 등도 가능하니다.


# 이미지 저장
# from PIL import Image
# im = Image.fromarray(final)
# im.save(new_img_path)

from skimage.io import imsave
imsave(new_img_path,img2)



#input image
img = cv2.imread(image_files[10],cv2.IMREAD_COLOR)
b, g, r = cv2.split(img)  # img파일을 b,g,r로 분리
img2 = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

plt.imshow(final)
plt.show()



load = np.load('/mnt/ssm/simul/1.npy', allow_pickle=True)



img = cv2.imread('/mnt/ssm/simul/1.png',cv2.IMREAD_COLOR)
b, g, r = cv2.split(img)  # img파일을 b,g,r로 분리
img2 = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

plt.imshow(img2)
plt.show()

scatter = plt.scatter(np.where(load==2)[0],np.where(load==2)[1])
plt.show()

output = model(proj1.float().cuda())

np.where(load==0)

# for i in np.arange(len(np.where(load==0)[0])):
#     load[np.where(load==0)[0][i],np.where(load==0)[1][i]] = 2

for i in np.arange(len(load)):
    for j in np.arange(len(load)):
        if load[i,j] == 0:
            load[i,j]=2





import torch
# w_dict = torch.load("/mnt/han/lidar-bonnetal/train/tasks/semantic/carla_pretrained" + "/backbone",
#                             map_location=lambda storage, loc: storage)

path = '/mnt/han/lidar-bonnetal/train/tasks/semantic/1028train'
path_append = "_train"
    # try backbone
try:
    w_dict = torch.load(path + "/initial_block" + path_append,
                        map_location=lambda storage, loc: storage)
    model.initial_block.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model initial_block weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/downsample1_0" + path_append,
                        map_location=lambda storage, loc: storage)
    model.downsample1_0.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model downsample1_0 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/regular1_1" + path_append,
                        map_location=lambda storage, loc: storage)
    model.regular1_1.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model regular1_1 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/regular1_2" + path_append,
                        map_location=lambda storage, loc: storage)
    model.regular1_2.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model regular1_2 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/regular1_3" + path_append,
                        map_location=lambda storage, loc: storage)
    model.regular1_3.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model regular1_3 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/regular1_4" + path_append,
                        map_location=lambda storage, loc: storage)
    model.regular1_4.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model regular1_4 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/downsample2_0" + path_append,
                        map_location=lambda storage, loc: storage)
    model.downsample2_0.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model downsample2_0 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/regular2_1" + path_append,
                        map_location=lambda storage, loc: storage)
    model.regular2_1.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model regular2_1 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/dilated2_2" + path_append,
                        map_location=lambda storage, loc: storage)
    model.dilated2_2.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model dilated2_2 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/asymmetric2_3" + path_append,
                        map_location=lambda storage, loc: storage)
    model.asymmetric2_3.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model asymmetric2_3 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/dilated2_4" + path_append,
                        map_location=lambda storage, loc: storage)
    model.dilated2_4.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model dilated2_4 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/regular2_5" + path_append,
                        map_location=lambda storage, loc: storage)
    model.regular2_5.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model regular2_5 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/dilated2_6" + path_append,
                        map_location=lambda storage, loc: storage)
    model.dilated2_6.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model dilated2_6 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/asymmetric2_7" + path_append,
                        map_location=lambda storage, loc: storage)
    model.asymmetric2_7.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model asymmetric2_7 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/dilated2_8" + path_append,
                        map_location=lambda storage, loc: storage)
    model.dilated2_8.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model dilated2_8 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/regular3_0" + path_append,
                        map_location=lambda storage, loc: storage)
    model.regular3_0.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model regular3_0 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/dilated3_1" + path_append,
                        map_location=lambda storage, loc: storage)
    model.dilated3_1.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model dilated3_1 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/asymmetric3_2" + path_append,
                        map_location=lambda storage, loc: storage)
    model.asymmetric3_2.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model asymmetric3_2 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/dilated3_3" + path_append,
                        map_location=lambda storage, loc: storage)
    model.dilated3_3.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model dilated3_3 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/regular3_4" + path_append,
                        map_location=lambda storage, loc: storage)
    model.regular3_4.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model regular3_4 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/dilated3_5" + path_append,
                        map_location=lambda storage, loc: storage)
    model.dilated3_5.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model dilated3_5 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/asymmetric3_6" + path_append,
                        map_location=lambda storage, loc: storage)
    model.asymmetric3_6.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model asymmetric3_6 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/dilated3_7" + path_append,
                        map_location=lambda storage, loc: storage)
    model.dilated3_7.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model dilated3_7 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/upsample4_0" + path_append,
                        map_location=lambda storage, loc: storage)
    model.upsample4_0.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model upsample4_0 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/regular4_1" + path_append,
                        map_location=lambda storage, loc: storage)
    model.regular4_1.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model regular4_1 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/regular4_2" + path_append,
                        map_location=lambda storage, loc: storage)
    model.regular4_2.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model regular4_2 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/upsample5_0" + path_append,
                        map_location=lambda storage, loc: storage)
    model.upsample5_0.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model upsample5_0 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/regular5_1" + path_append,
                        map_location=lambda storage, loc: storage)
    model.regular5_1.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model regular5_1 weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
try:
    w_dict = torch.load(path + "/transposed_conv" + path_append,
                        map_location=lambda storage, loc: storage)
    model.transposed_conv.load_state_dict(w_dict, strict=False)
    print("Successfully loaded model transposed_conv weights")
except Exception as e:
    print()
    print("Couldn't load backbone, using random weights. Error: ", e)
    if strict:
        print("I'm in strict mode and failure to load weights blows me up :)")
        raise e
