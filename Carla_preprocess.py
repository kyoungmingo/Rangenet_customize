import os
import numpy as np
import torch
from torch.utils.data import Dataset
from common.laserscan_c import LaserScan, SemLaserScan
from skimage import transform
import cv2
from torchvision import transforms, utils

# 확장자
EXTENSIONS_IMAGE = ['.png']
EXTENSIONS_SCAN = ['.pcd']

# file load
def is_image(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_IMAGE)

def is_pcd(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

label_path = os.path.join("/mnt/kkm/carla/1014_Logging/pcd_seg")
image_path = os.path.join("/mnt/kkm/carla/1014_Logging/img_seg")
image_path2 = os.path.join("/mnt/kkm/carla/1014_Logging/img")

label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
  os.path.expanduser(label_path)) for f in fn if is_pcd(f)]
image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
  os.path.expanduser(image_path)) for f in fn if is_image(f)]
image_files2 = [os.path.join(dp, f) for dp, dn, fn in os.walk(
  os.path.expanduser(image_path2)) for f in fn if is_image(f)]

# scan_files.sort()
label_files.sort()
image_files.sort()

#[0]
image_file = image_files[0]
label_file = label_files[1]
scan_file = scan_files[0]

# if all goes well, open label
label = np.fromfile(label_file, dtype=np.int32)
label = label.reshape((-1))

sem_label = label & 0xFFFF  # semantic label in lower half
inst_label = label >> 16  # instance id in upper half

# only map colors to labels that exist
mask = self.proj_idx >= 0

# semantics
self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]

#image load


# input image
img = cv2.imread(image_file, cv2.IMREAD_COLOR)
b, g, r = cv2.split(img)  # img파일을 b,g,r로 분리
img2 = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge
img2 = np.float32(img2)
img2[(img2==0)] = 100

# the three channels
a = np.array([0, 0, 0])
b = np.array([70, 70, 70])
c = np.array([190, 153, 153])
d = np.array([250, 170, 160])
e = np.array([220, 20, 60])
f = np.array([153, 153, 153])
g = np.array([157, 234, 50])
h = np.array([128, 64, 128])
i = np.array([244, 35, 232])
j = np.array([107, 142, 35])
k = np.array([0, 0, 142])
l = np.array([102, 102, 156])
m = np.array([220, 220, 0])
n = np.array([70, 130, 180])
o = np.array([81, 0, 81])
p = np.array([150, 100, 100])



label_seg = np.zeros((img.shape[:2]), dtype=np.int)
label_seg.fill(-1)
label_seg[(img==a).all(axis=2)] = 0
label_seg[(img==b).all(axis=2)] = 1
label_seg[(img==c).all(axis=2)] = 2
label_seg[(img==d).all(axis=2)] = 3
label_seg[(img==e).all(axis=2)] = 4
label_seg[(img==f).all(axis=2)] = 5
label_seg[(img==g).all(axis=2)] = 6
label_seg[(img==h).all(axis=2)] = 7
label_seg[(img==i).all(axis=2)] = 8
label_seg[(img==j).all(axis=2)] = 9
label_seg[(img==k).all(axis=2)] = 10
label_seg[(img==l).all(axis=2)] = 11
label_seg[(img==m).all(axis=2)] = 12
label_seg[(img==n).all(axis=2)] = 13
label_seg[(img==o).all(axis=2)] = 14
label_seg[(img==p).all(axis=2)] = 15


import matplotlib.pyplot as plt

plt.imshow(img2[:,:,0])
plt.show()

np.unique(img)



#############1015 Projection Test#######################
import yaml
import torch

ARCH = yaml.safe_load(open("/mnt/han/lidar-bonnetal/train/tasks/semantic/carla_pretrained/arch_cfg.yaml", 'r'))
DATA = yaml.safe_load(open("/mnt/han/lidar-bonnetal/train/tasks/semantic/carla_pretrained/data_cfg.yaml", 'r'))
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
sensor_img_H = sensor["img_prop"]["height"]
sensor_img_W = sensor["img_prop"]["width"]
sensor_img_means = torch.tensor(sensor["img_means"],
                                     dtype=torch.float)
sensor_img_stds = torch.tensor(sensor["img_stds"],
                                    dtype=torch.float)
sensor_fov_up = sensor["fov_up"]
sensor_fov_down = sensor["fov_down"]

import os
import numpy as np
import torch
from torch.utils.data import Dataset
# from common.laserscan_c import LaserScan, SemLaserScan
from skimage import transform
import cv2
from torchvision import transforms, utils

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']
EXTENSIONS_IMAGE = ['.png']

def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

def is_image(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_IMAGE)

fscan_files = []
flabel_files = []
fimage_files = []

scan_path = os.path.join("/mnt/kkm/cdataset/sequences", "00",
                               "velodyne")
label_path = os.path.join("/mnt/kkm/cdataset/sequences", "00",
                        "labels")
image_path = os.path.join("/mnt/kkm/cdataset/img", "00")

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
# from PIL import  Image

for i in np.arange(len(fimage_files)):


    scan_file = fscan_files[i]
    image_file = fimage_files[i]
    if gt:
      label_file = flabel_files[i]

    # open a semantic laserscan
    # if gt:
    #   scan = SemLaserScan(color_map,
    #                       project=True,
    #                       H=sensor_img_H,
    #                       W=sensor_img_W,
    #                       fov_up=sensor_fov_up,
    #                       fov_down=sensor_fov_down)
    # else:
    #   scan = LaserScan(project=True,
    #                    H=sensor_img_H,
    #                    W=sensor_img_W,
    #                    fov_up=sensor_fov_up,
    #                    fov_down=sensor_fov_down)

    # open and obtain scan
    scan.open_scan(scan_file)
    if gt:
        scan.open_label(label_file)
      # map unused classes to used classes (also for projection)
      #map함수는 Semantic kitti 내장 map 함수를 사용해야함. 아니면, python 기본 내장 map함수로 다음 진행이 이뤄지지 않음.
        scan.sem_label = map(scan.sem_label, learning_map)
        scan.proj_sem_label = map(scan.proj_sem_label, learning_map)

    #input image
    # img = cv2.imread(image_file,cv2.IMREAD_COLOR)
    # b, g, r = cv2.split(img)  # img파일을 b,g,r로 분리
    # img2 = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

    label = map(scan.proj_sem_label, learning_map_inv)
    final = map(label, color_map)

    path_norm = os.path.normpath(image_file)
    path_split = path_norm.split(os.sep)
    path_seq = path_split[-3]
    path_name = path_split[-1]
    plt.imsave('/mnt/kkm/cdataset/gtmask/00/' + path_name, final)








import matplotlib.pyplot as plt

plt.imshow(final)
plt.show()

img = cv2.imread(image_file,cv2.IMREAD_COLOR)
b, g, r = cv2.split(img)  # img파일을 b,g,r로 분리
img2 = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

plt.imshow(img2)
plt.show()

dst = cv2.resize(img2, dsize=(256, 200), interpolation=cv2.INTER_AREA)

plt.imshow(dst)
plt.show()

##point가 있는 mask 이미지에 씌우기##

# laser parameters
fov_up = scan.proj_fov_up / 180 * np.pi      # field of view up in rad
fov_down = scan.proj_fov_down / 180 * np.pi  # field of view down in rad
# fov_up = self.proj_fov_up / 180 * np.pi  # field of view up in rad
# fov_down = self.proj_fov_down /180 * np.pi  # field of view down in rad
fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

# get depth of all points
depth = np.linalg.norm(scan.points, 2, axis=1)

# get scan components
scan_x = scan.points[:, 0]
scan_y = scan.points[:, 1]
scan_z = scan.points[:, 2]

# get angles of all points
yaw = -np.arctan2(scan_y, scan_x)
pitch = np.arcsin(scan_z / depth)

# get projections in image coords
proj_x = 0.5 * (yaw /(110.0 / 360.0 * np.pi) + 1.0)          # in [0.0, 1.0]
proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

# scale to image size using angular resolution
proj_x *= scan.proj_W                              # in [0.0, W]
proj_y *= scan.proj_H                              # in [0.0, H]

# round and clamp for use as index
proj_x = np.floor(proj_x)
proj_x = np.minimum(scan.proj_W - 1, proj_x)
proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
scan.proj_x = np.copy(proj_x)  # store a copy in orig order

proj_y = np.floor(proj_y)
proj_y = np.minimum(scan.proj_H - 1, proj_y)
proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
scan.proj_y = np.copy(proj_y)  # stope a copy in original order

# copy of depth in original order
scan.unproj_range = np.copy(depth)

# order in decreasing depth
indices = np.arange(depth.shape[0])
order = np.argsort(depth)[::-1]
depth = depth[order]
indices = indices[order]
points = scan.points[order]
remission = scan.remissions[order]
proj_y = proj_y[order]
proj_x = proj_x[order]

# assing to images
scan.proj_range[proj_y, proj_x] = depth
scan.proj_xyz[proj_y, proj_x] = points
scan.proj_remission[proj_y, proj_x] = remission
scan.proj_idx[proj_y, proj_x] = indices
scan.proj_mask = (scan.proj_idx > 0).astype(np.int32)













EXTENSIONS_SCAN = ['.pcd']

def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

scan_files = []

scan_path = os.path.join("/mnt/kkm/carla/1014_Logging/pcd_seg")

scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)]

scan_files.sort()

scan_file = scan_files[0]

scan = np.fromfile(scan_file, dtype=np.float32)
scan = scan.reshape((-1, 4))

# put in attribute
points = scan[:, 0:3]  # get xyz
remissions = scan[:, 3]  # get remission




#####Lane Mark Detection#####

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from common.laserscan_c import LaserScan, SemLaserScan
from skimage import transform
import cv2
from torchvision import transforms, utils

# 확장자
EXTENSIONS_IMAGE = ['.png']
EXTENSIONS_SCAN = ['.pcd']

# file load
def is_image(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_IMAGE)


image_path = os.path.join("/mnt/lyh/sample/anno")
label_path = os.path.join("/mnt/lyh/pic/bus/gt")
image_path = os.path.join("/mnt/lyh/pic/bus")


image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
  os.path.expanduser(image_path)) for f in fn if is_image(f)]
label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
  os.path.expanduser(label_path)) for f in fn if is_image(f)]




image_files.sort()

#[0]
image_file = image_files[1]
label_file = label_files[0]

# input image
img = cv2.imread(image_file, cv2.IMREAD_COLOR)
b, g, r = cv2.split(img)  # img파일을 b,g,r로 분리
img2 = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge
img2 = np.float32(img2)

# lab = cv2.imread(label_file)


# the three channels
a = np.array([0, 0, 0])
b = np.array([70, 70, 70])
c = np.array([190, 153, 153])
d = np.array([250, 170, 160])
e = np.array([220, 20, 60])
f = np.array([153, 153, 153])
g = np.array([157, 234, 50])
h = np.array([128, 64, 128])
i = np.array([244, 35, 232])
j = np.array([107, 142, 35])
k = np.array([0, 0, 142])
l = np.array([102, 102, 156])
m = np.array([220, 220, 0])
n = np.array([70, 130, 180])
o = np.array([81, 0, 81])
p = np.array([150, 100, 100])



label_seg = np.zeros((img.shape[:2]), dtype=np.int)
label_seg.fill(-1)
label_seg[(img==a).all(axis=2)] = 0
label_seg[(img==b).all(axis=2)] = 1
label_seg[(img==c).all(axis=2)] = 2
label_seg[(img==d).all(axis=2)] = 3
label_seg[(img==e).all(axis=2)] = 4
label_seg[(img==f).all(axis=2)] = 5
label_seg[(img==g).all(axis=2)] = 6
label_seg[(img==h).all(axis=2)] = 7
label_seg[(img==i).all(axis=2)] = 8
label_seg[(img==j).all(axis=2)] = 9
label_seg[(img==k).all(axis=2)] = 10
label_seg[(img==l).all(axis=2)] = 11
label_seg[(img==m).all(axis=2)] = 12
label_seg[(img==n).all(axis=2)] = 13
label_seg[(img==o).all(axis=2)] = 14
label_seg[(img==p).all(axis=2)] = 15


import matplotlib.pyplot as plt

plt.imshow(img2)
           # , cmap='gray')
plt.show()

np.unique(img2)


# input image
img = cv2.imread(image_files[3], cv2.IMREAD_GRAYSCALE)
# b, g, r = cv2.split(img)  # img파일을 b,g,r로 분리
# img2 = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge
# img2 = np.float32(img)
# input image
lab = cv2.imread(image_files[2], cv2.IMREAD_GRAYSCALE)
# b, g, r = cv2.split(img)  # img파일을 b,g,r로 분리
# lab2 = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge
# lab2 = np.float32(lab)
# input image
ins = cv2.imread(image_files[1], cv2.IMREAD_GRAYSCALE)
# b, g, r = cv2.split(img)  # img파일을 b,g,r로 분리
# ins2 = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge
# ins2 = np.float32(ins)
# input image
iimg = cv2.imread(image_files[0], cv2.IMREAD_COLOR)
b, g, r = cv2.split(iimg)  # img파일을 b,g,r로 분리
iimg2 = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge
# iimg2 = np.float32(iimg2)
plt.imshow(iimg2)
plt.show()

len(np.unique(img2))
len(np.unique(lab2))
len(np.unique(ins2))
len(np.unique(iimg2[:,:,2]))

# input image
img = cv2.imread(image_file, cv2.IMREAD_COLOR)
b, g, r = cv2.split(img)  # img파일을 b,g,r로 분리
img2 = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge
img2 = np.float32(img2)

plt.imshow(img2)
plt.show()

Label = ()
labels = [ (  0,  0,  0)
    , (111, 74,  0)
    , ( 81,  0, 81)
    , (128, 64,128)
    , (244, 35,232)
    , (250,170,160)
    , (230,150,140)
    , ( 70, 70, 70)
    , (102,102,156)
    , (190,153,153)
    , (180,165,180)
    , (150,100,100)
    , (150,120, 90)
    , (153,153,153)
    , (153,153,153)
    , (250,170, 30)
    , (220,220,  0)
    , (107,142, 35)
    , (152,251,152)
    , ( 70,130,180)
    , (220, 20, 60)
    , (255,  0,  0)
    , (  0,  0,142)
    , (  0,  0, 70)
    , (  0, 60,100)
    , (  0,  0, 90)
    , (  0,  0,110)
    , (  0, 80,100)
    , (  0,  0,230)
    , (119, 11, 32)
    , (  0,  0,142)
]

label_seg = np.zeros((img2.shape[:2]), dtype=np.int)
label_seg.fill(-1)
for i in np.arange(len(labels)):
    label_seg[(img2==np.array(labels[i])).all(axis=2)] = np.arange(len(labels))[i]


proj = torch.from_numpy(img2)
proj = proj.permute(2,0,1)
input=proj.unsqueeze(0)
# model.cuda()
output=model(input)
argmax = output.argmax(dim=1)
argmax2 = argmax.numpy()

pred1 = np.zeros([1,534,800], dtype=np.int)
pred2 = np.zeros([1,534,800], dtype=np.int)
pred3 = np.zeros([1,534,800], dtype=np.int)

pred1.fill(-1)
pred2.fill(-1)
pred3.fill(-1)


for i in np.arange(len(labels)):
    pred1[argmax2 == np.arange(len(labels))[i]] = np.array(labels[i])[0]
    pred2[argmax2 == np.arange(len(labels))[i]] = np.array(labels[i])[1]
    pred3[argmax2 == np.arange(len(labels))[i]] = np.array(labels[i])[2]


x = np.append(pred1,pred2,axis=0)
x = np.append(x,pred3,axis=0)



xx=np.transpose(x,(1,2,0))

plt.imshow(xx)
# plt.imshow(img2)
plt.show()

cv2.imshow('origin1', xx)