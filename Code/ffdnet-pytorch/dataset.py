"""
Dataset related functions

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import os.path
import random
import glob
import numpy as np
import cv2
import h5py
import torch
import torch.utils.data as udata
from utils import data_augmentation, normalize
import tqdm

def img_to_patches(img, win, stride=1):
    r"""Converts an image to an array of patches.

    Args:
        img: a numpy array containing a CxHxW RGB (C=3) or grayscale (C=1)
            image
        win: size of the output patches
        stride: int. stride
    """
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    total_pat_num = patch.shape[1] * patch.shape[2]
    res = np.zeros([endc, win*win, total_pat_num], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw-win+i+1:stride, j:endh-win+j+1:stride]
            res[:, k, :] = np.array(patch[:]).reshape(endc, total_pat_num)
            k = k + 1
    return res.reshape([endc, win, win, total_pat_num])

def prepare_data(data_path, \
                 val_data_path, \
                 patch_size, \
                 stride, \
                 max_num_patches=None, \
                 aug_times=1, \
                 gray_mode=False):
    r"""Builds the training and validations datasets by scanning the
    corresponding directories for images and extracting	patches from them.

    Args:
        data_path: path containing the training image dataset
        val_data_path: path containing the validation image dataset
        patch_size: size of the patches to extract from the images
        stride: size of stride to extract patches
        stride: size of stride to extract patches
        max_num_patches: maximum number of patches to extract
        aug_times: number of times to augment the available data minus one
        gray_mode: build the databases composed of grayscale patches
    """
    # training database
    print('> Training database')
    scales = [1, 0.9, 0.8]
    types = ('**/*.bmp', '**/*.png')
    files = []
    for tp in types:
        files.extend(glob.glob(os.path.join(data_path, tp), recursive=True))
    files.sort()
 
    originalFiles = []
    # original files have _original in their name
    for file in files:
        if "_original" in file:
            originalFiles.append(file)
            files.remove(file)
    
    print("FILES:")
    for file in files:
        print(file)
    print("\tNumber of files: {}".format(len(files)))
    
    print("ORIGINALS:")
    for file in originalFiles:
        print(file)
    print("\tNumber of original files: {}".format(len(originalFiles)))
    

    if gray_mode:
        traindbf = 'train_gray.h5'
        trainorigdbf = 'train_gray_original.h5'
        valdbf = 'val_gray.h5'
        valorigdbf = 'val_gray_original.h5'
    else:
        traindbf = 'train_rgb.h5'
        trainorigdbf = 'train_rgb_original.h5'
        valdbf = 'val_rgb.h5'
        valorigdbf = 'val_rgb_original.h5'

    if max_num_patches is None:
        max_num_patches = 5000000
        print("\tMaximum number of patches not set")
    else:
        print("\tMaximum number of patches set to {}".format(max_num_patches))
    train_num = 0
    i = 0
    with h5py.File(trainorigdbf, 'w') as h5fOrig:
        with h5py.File(traindbf, 'w') as h5f:
            for file in tqdm.tqdm(files):
                if (train_num >= max_num_patches):
                    break
                imgor = cv2.imread(file)
                # get corresponding original image, based on the prefix before the _original
                
                imgorOriginal = None
                for originalFile in originalFiles:
                    # print(file.split("/")[-1].split("_")[0].split(".")[0])
                    # print(originalFile.split("/")[-1])
                    # print()
                    if file.split("/")[-1].split("_")[0].split(".")[0] in originalFile.split("/")[-1]:
                        imgorOriginal = cv2.imread(originalFile)
                        break
                    
                if imgorOriginal is None:
                    print("Error: Could not find original image for {}".format(file))
                    continue
                
                # h, w, c = img.shape
                for sca in scales:
                    img = cv2.resize(imgor, (0, 0), fx=sca, fy=sca, \
                                    interpolation=cv2.INTER_CUBIC)
                    imgOriginal = cv2.resize(imgorOriginal, (0, 0), fx=sca, fy=sca, \
                                    interpolation=cv2.INTER_CUBIC)
                    
                    if not gray_mode:
                        # CxHxW RGB image
                        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
                        imgOriginal = (cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
                    else:
                        # CxHxW grayscale image (C=1)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = np.expand_dims(img, 0)
                        imgOriginal = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
                        imgOriginal = np.expand_dims(imgOriginal, 0)
                    img = normalize(img)
                    imgOriginal = normalize(imgOriginal)
                    patches = img_to_patches(img, win=patch_size, stride=stride)
                    patchesOriginal = img_to_patches(imgOriginal, win=patch_size, stride=stride)
                    # print("\tfile: %s scale %.1f # samples: %d" % \
                    #     (file, sca, patches.shape[3]*aug_times))
                    for nx in range(patches.shape[3]):
                        n = np.random.randint(0, 7)
                        data = data_augmentation(patches[:, :, :, nx].copy(), \
                                n)
                        dataOriginal = data_augmentation(patchesOriginal[:, :, :, nx].copy(), \
                                n)
                        h5f.create_dataset(str(train_num), data=data)
                        h5fOrig.create_dataset(str(train_num) + "_original", data=dataOriginal)
                        train_num += 1
                        for mx in range(aug_times-1):
                            n = np.random.randint(1, 4)
                            data_aug = data_augmentation(data, n)
                            dataOriginal_aug = data_augmentation(dataOriginal, n)
                            h5f.create_dataset(str(train_num)+"_aug_%d" % (mx+1), data=data_aug)
                            h5fOrig.create_dataset(str(train_num)+"_aug_%d_original" % (mx+1), data=dataOriginal_aug)
                            train_num += 1
                i += 1

    # validation database
    print('\n> Validation database')
    files = []
    for tp in types:
        files.extend(glob.glob(os.path.join(val_data_path, tp), recursive=True))
    files.sort()
    
    # original files have _original in their name
    originalFiles = []
    for file in files:
        if "_original" in file:
            originalFiles.append(file)
            files.remove(file)
            
    print("VAL FILES:")
    for file in files:
        print(file)
    print("\tNumber of files: {}".format(len(files)))
    
    print("VAL ORIGINALS:")
    for file in originalFiles:
        print(file)
    print("\tNumber of original files: {}".format(len(originalFiles)))
    
    h5f = h5py.File(valdbf, 'w')
    h5fOrig = h5py.File(valorigdbf, 'w')
    val_num = 0
    for i, item in tqdm.tqdm(enumerate(files)):
        # print("\tfile: %s" % item)
        img = cv2.imread(item)
        
        # get corresponding original image, based on the prefix before the _original
        itemOriginal = None
        for originalFile in originalFiles:
            if item.split("/")[-1].split("_")[0].split(".")[0] in originalFile.split("/")[-1]:
                itemOriginal = cv2.imread(originalFile)
                break
        
        if itemOriginal is None:
            print("Error: Could not find original image for {}".format(item))
            continue
        
        imgOriginal = cv2.imread(itemOriginal)
        if not gray_mode:
            # C. H. W, RGB image
            img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
            imgOriginal = (cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
        else:
            # C, H, W grayscale image (C=1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, 0)
            imgOriginal = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
            imgOriginal = np.expand_dims(imgOriginal, 0)
        img = normalize(img)
        imgOriginal = normalize(imgOriginal)
        h5f.create_dataset(str(val_num), data=img)
        h5fOrig.create_dataset(str(val_num) + "_original", data=imgOriginal)
        val_num += 1
    h5f.close()
    h5fOrig.close()

    print('\n> Total')
    print('\ttraining set, # samples %d' % train_num)
    print('\tvalidation set, # samples %d\n' % val_num)

class Dataset(udata.Dataset):
    r"""Implements torch.utils.data.Dataset
    """
    def __init__(self, train=True, gray_mode=False, shuffle=False):
        super(Dataset, self).__init__()
        self.train = train
        self.gray_mode = gray_mode
        if not self.gray_mode:
            self.traindbf = 'train_rgb.h5'
            self.valdbf = 'val_rgb.h5'
        else:
            self.traindbf = 'train_gray.h5'
            self.valdbf = 'val_gray.h5'

        if self.train:
            h5f = h5py.File(self.traindbf, 'r')
        else:
            h5f = h5py.File(self.valdbf, 'r')
        self.keys = list(h5f.keys())
        if shuffle:
            random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File(self.traindbf, 'r')
        else:
            h5f = h5py.File(self.valdbf, 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)
