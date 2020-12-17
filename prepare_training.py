#Carla data 기반
#Class 별, point 비율 정리

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

def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

sequences = ['0','1','2','3','4']

fscan_files = []
flabel_files = []

for seq in sequences:
    # to string
    seq = '{0:02d}'.format(int(seq))

    print("parsing seq {}".format(seq))

    # get paths for each
    scan_path = os.path.join('/mnt/kkm/cdataset/sequences', seq, "velodyne")
    label_path = os.path.join('/mnt/kkm/cdataset/sequences', seq, "labels")

    # get files
    scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
    label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(label_path)) for f in fn if is_label(f)]

    assert(len(scan_files) == len(label_files))

    # extend list
    fscan_files.extend(scan_files)
    flabel_files.extend(label_files)

    # sort for correspondance
    fscan_files.sort()
    flabel_files.sort()

  # 0 :  "unlabeled"
  # 1 :  "building"
  # 2 :  "fence"
  # 3 :  "other-structure"
  # 4 :  "person"
  # 5 :  "pole"
  # 6 :  "lane-marking"
  # 7 :  "road"
  # 8 :  "sidewalk"
  # 9 :  "vegetation"
  # 10:  "car"
  # 11:  "wall"
  # 12:  "traffic-sign"
total_point =0
point_0 =0
point_1 =0
point_2 =0
point_3 =0
point_4 =0
point_5 =0
point_6 =0
point_7 =0
point_8 =0
point_9 =0
point_10 =0
point_11 =0
point_12 =0

for i in range(len(flabel_files)):
    # scan_file = fscan_files[0]
    label_file = flabel_files[i]
    # scan = np.fromfile(scan_file, dtype=np.float32)
    # scan = scan.reshape((-1, 4))
    label = np.fromfile(label_file, dtype=np.int32)
    label = label.reshape((-1))

    total_point += label.shape[0]
    point_0 += np.sum(label==0)
    point_1 += np.sum(label==1)
    point_2 += np.sum(label==2)
    point_3 += np.sum(label==3)
    point_4 += np.sum(label==4)
    point_5 += np.sum(label==5)
    point_6 += np.sum(label==6)
    point_7 += np.sum(label==7)
    point_8 += np.sum(label==8)
    point_9 += np.sum(label==9)
    point_10 += np.sum(label==10)
    point_11 += np.sum(label==11)
    point_12 += np.sum(label==12)

prob0 = point_0/total_point
prob1 = point_1/total_point
prob2 = point_2/total_point
prob3 = point_3/total_point
prob4 = point_4/total_point
prob5 = point_5/total_point
prob6 = point_6/total_point
prob7 = point_7/total_point
prob8 = point_8/total_point
prob9 = point_9/total_point
prob10 = point_10/total_point
prob11 = point_11/total_point
prob12 = point_12/total_point