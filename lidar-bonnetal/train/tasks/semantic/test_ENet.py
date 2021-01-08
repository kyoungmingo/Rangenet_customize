from modules.ENet import *
import os
import cv2
import torch
import numpy as np


EXTENSIONS_IMAGE = ['.png']

def is_image(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_IMAGE)

image_path = "/mnt/nail_data/img_resize2"

image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
      os.path.expanduser(image_path)) for f in fn if is_image(f)]

img = cv2.imread(image_files[5],cv2.IMREAD_COLOR)
b, g, r = cv2.split(img)  # img파일을 b,g,r로 분리
img2 = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge
img2 = np.float32(img2)
input = torch.from_numpy(img2)
input = input.permute(2,0,1)
input=input.unsqueeze(0)

model = ENet(3)

model(input)

