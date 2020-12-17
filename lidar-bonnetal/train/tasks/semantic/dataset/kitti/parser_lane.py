import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from torchvision import transforms, utils


EXTENSIONS_IMAGE = ['.png']

def is_image(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_IMAGE)


def convert_image(lab):
  labels = [(0, 0, 0)
    , (111, 74, 0)
    , (81, 0, 81)
    , (128, 64, 128)
    , (244, 35, 232)
    , (250, 170, 160)
    , (230, 150, 140)
    , (70, 70, 70)
    , (102, 102, 156)
    , (190, 153, 153)
    , (180, 165, 180)
    , (150, 100, 100)
    , (150, 120, 90)
    , (153, 153, 153)
    , (153, 153, 153)
    , (250, 170, 30)
    , (220, 220, 0)
    , (107, 142, 35)
    , (152, 251, 152)
    , (70, 130, 180)
    , (220, 20, 60)
    , (255, 0, 0)
    , (0, 0, 142)
    , (0, 0, 70)
    , (0, 60, 100)
    , (0, 0, 90)
    , (0, 0, 110)
    , (0, 80, 100)
    , (0, 0, 230)
    , (119, 11, 32)
    , (0, 0, 142)
            ]

  label_seg = np.zeros((lab.shape[:2]), dtype=np.int)
  label_seg.fill(-1)
  for i in np.arange(len(labels)):
    label_seg[(lab == np.array(labels[i])).all(axis=2)] = np.arange(len(labels))[i]
  return label_seg

class SemanticKitti(Dataset):

  def __init__(self, root,    # directory where data is
               sequences,     # sequences for this data (e.g. [1,3,4,6])
               labels,        # label dict: (e.g 10: "car")
               gt=True):            # send ground truth?

    # save deats
    self.sequences = sequences
    self.labels = labels
    self.gt = gt
    self.aug_f = transforms.Compose([transforms.ToTensor()])

    # get number of classes (can't be len(self.learning_map) because there
    # are multiple repeated entries, so the number that matters is how many
    # there are for the xentropy)
    self.nclasses = 4

    # placeholder for filenames
    self.label_files = []
    self.image_files = []

    # fill in with names, checking that all sequences are complete
    for seq in self.sequences:
      # to string
      seq = '{0:02d}'.format(int(seq))

      print("parsing seq {}".format(seq))

      # get paths for each
      # label_path = os.path.join("/mnt/lyh/sample/anno")
      # image_path = os.path.join("/mnt/lyh/sample/origin")
      label_path = os.path.join("/mnt/lyh/pic/bus/gt")
      image_path = os.path.join("/mnt/lyh/pic/bus/origin")



      # get files

      label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(label_path)) for f in fn if is_image(f)]
      image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(image_path)) for f in fn if is_image(f)]

      label_files.sort()
      image_files.sort()


      # extend list

      self.label_files.extend(label_files)
      self.image_files.extend(image_files)

    # sort for correspondance

    self.label_files.sort()
    self.image_files.sort()

  def __getitem__(self, index):
    # get item in tensor shape
    image_file = self.image_files[index]
    if self.gt:
      label_file = self.label_files[index]

    #input image
    img = cv2.imread(image_file,cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)  # img파일을 b,g,r로 분리
    img2 = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge
    img2 = np.float32(img2)

    lab = cv2.imread(label_file)
    # b, g, r = cv2.split(lab)  # img파일을 b,g,r로 분리
    # lab2 = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge
    # load = convert_image(lab2)
    load = lab[:,:,0]
    load[load==0] = 1
    load[load == 2] = 2
    load[load == 255] = 3
    load = np.int32(load)

    # make a tensor of the uncompressed data (with the max num points)
    unproj_n_points = []
    unproj_xyz = []
    unproj_range = []
    unproj_remissions = []
    if self.gt:
      unproj_labels = []
    else:
      unproj_labels = []

    # get points and labels
    proj_range = []
    proj_xyz = []
    proj_remission = []
    proj_mask = torch.from_numpy(load)
    if self.gt:
      proj_labels = torch.from_numpy(load).clone()
    else:
      proj_labels = []
    proj_x = []
    proj_y = []
    proj = torch.from_numpy(img2)
    proj = proj.permute(2,0,1)

    # get name and sequence
    path_norm = os.path.normpath(image_file)
    path_split = path_norm.split(os.sep)
    path_seq = path_split[-3]
    path_name = path_split[-1].replace(".png", ".npy")


    # return
    return proj, proj_mask, proj_labels, unproj_labels, path_seq, path_name, proj_x, proj_y, proj_range, unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points

  def __len__(self):
    return len(self.image_files)

  @staticmethod
  def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]


class Parser():
  # standard conv, BN, relu
  def __init__(self,
               root,              # directory for data
               train_sequences,   # sequences to train
               valid_sequences,   # sequences to validate.
               test_sequences,    # sequences to test (if none, don't get)
               labels,            # labels in data
               color_map,         # color for each label
               learning_map,      # mapping for training labels
               learning_map_inv,  # recover labels from xentropy
               sensor,            # sensor to use
               max_points,        # max points in each scan in entire dataset
               batch_size,        # batch size for train and val
               workers,           # threads to load data
               gt=True,           # get gt?
               shuffle_train=True):  # shuffle training set?
    super(Parser, self).__init__()

    # if I am training, get the dataset
    self.root = root
    self.train_sequences = train_sequences
    self.valid_sequences = valid_sequences
    self.test_sequences = test_sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.max_points = max_points
    self.batch_size = batch_size
    self.workers = workers
    self.gt = gt
    self.shuffle_train = shuffle_train

    # number of classes that matters is the one for xentropy
    self.nclasses = 4

    # Data loading code
    self.train_dataset = SemanticKitti(root=self.root,
                                       sequences=self.train_sequences,
                                       labels=self.labels,
                                       gt=self.gt)

    self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=self.shuffle_train,
                                                   num_workers=self.workers,
                                                   pin_memory=True,
                                                   drop_last=True)
    assert len(self.trainloader) > 0
    self.trainiter = iter(self.trainloader)

    self.valid_dataset = SemanticKitti(root=self.root,
                                       sequences=self.valid_sequences,
                                       labels=self.labels,
                                       gt=self.gt)

    self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers=self.workers,
                                                   pin_memory=True,
                                                   drop_last=True)
    assert len(self.validloader) > 0
    self.validiter = iter(self.validloader)

    if self.test_sequences:
      self.test_dataset = SemanticKitti(root=self.root,
                                        sequences=self.test_sequences,
                                        labels=self.labels,
                                        gt=False)

      self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=False,
                                                    num_workers=self.workers,
                                                    pin_memory=True,
                                                    drop_last=True)
      assert len(self.testloader) > 0
      self.testiter = iter(self.testloader)

  def get_train_batch(self):
    scans = self.trainiter.next()
    return scans

  def get_train_set(self):
    return self.trainloader

  def get_valid_batch(self):
    scans = self.validiter.next()
    return scans

  def get_valid_set(self):
    return self.validloader

  def get_test_batch(self):
    scans = self.testiter.next()
    return scans

  def get_test_set(self):
    return self.testloader

  def get_train_size(self):
    return len(self.trainloader)

  def get_valid_size(self):
    return len(self.validloader)

  def get_test_size(self):
    return len(self.testloader)

  def get_n_classes(self):
    return self.nclasses

  def get_original_class_string(self, idx):
    return self.labels[idx]

  def get_xentropy_class_string(self, idx):
    return self.labels[self.learning_map_inv[idx]]

  def to_original(self, label):
    # put label in original values
    return SemanticKitti.map(label, self.learning_map_inv)

  def to_xentropy(self, label):
    # put label in xentropy values
    return SemanticKitti.map(label, self.learning_map)

  def to_color(self, label):
    # put label in original values
    label = SemanticKitti.map(label, self.learning_map_inv)
    # put label in color
    return SemanticKitti.map(label, self.color_map)