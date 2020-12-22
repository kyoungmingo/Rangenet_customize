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
import matplotlib.pyplot as plt

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']
EXTENSIONS_IMAGE = ['.png']

def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

def is_image(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_IMAGE)

# open a semantic laserscan
#laserscan.py 실행
scan = SemLaserScan(color_map,
                  project=True,
                  H=sensor_img_H,
                  W=sensor_img_W,
                  fov_up=sensor_fov_up,
                  fov_down=sensor_fov_down)

sequences = [1,2,3,4]

for seq in sequences:
    # to string
    seq = '{0:02d}'.format(int(seq))

    print("parsing seq {}".format(seq))


    fscan_files = []
    flabel_files = []
    fimage_files = []

    scan_path = os.path.join("/mnt/kkm/cdataset/sequences", seq,
                                   "velodyne")
    label_path = os.path.join("/mnt/kkm/cdataset/sequences", seq,
                            "labels")
    image_path = os.path.join("/mnt/kkm/cdataset/img", seq)

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

#image file 기준으로 공통점 정의해야함.
    for i in np.arange(len(fimage_files)):


        scan_file = fscan_files[i]
        image_file = fimage_files[i]
        label_file = flabel_files[i]

        # open and obtain scan
        scan.open_scan(scan_file)
        if gt:
            scan.open_label(label_file)
          # map unused classes to used classes (also for projection)
          #map함수는 Semantic kitti 내장 map 함수를 사용해야함. 아니면, python 기본 내장 map함수로 다음 진행이 이뤄지지 않음.
          #parser.py >> map 함수 정의
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
        # path_seq = path_split[-3]
        path_name = path_split[-1]
        gtroot=os.path.join('/mnt/kkm/cdataset/gtmask',seq,path_name )
        plt.imsave(gtroot, final)
