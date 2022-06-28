import torch
import numpy as np
import cv2
import os
import pathlib
from subprocess import check_output
from PIL import Image


'''
# --------------------------------------------
# get file paths; optimized for quick turnaround
# --------------------------------------------
'''

def list_files_in_dir(dir):
    list_dir = str(pathlib.Path(__file__).parent.absolute()) + '/listdir'
    file_names = check_output([list_dir, dir])
    fnames = file_names.decode().strip().split('\n')
    return fnames

def get_file_paths(root, cache_dir):
    # a file will contain list of filepaths
    if os.path.isfile(root):
        cache_file_path = root
    else:
        # if cached
        cache_filename = "_".join(root.split('/')) + '.txt'
        cache_file_path = os.path.join(cache_dir, cache_filename)

    if os.path.isfile(cache_file_path):
        with open(cache_file_path, 'r', encoding='utf-8') as fi:
            filenames = fi.readlines()
        filenames = [x.strip() for x in filenames]
    else:
        filenames = list_files_in_dir(root)
        try:
            with open(cache_file_path, 'w', encoding='utf-8') as fi:
                fi.write('\n'.join(filenames))
        except Exception as e:
            # clean by deleting the file
            os.remove(cache_file_path)
            print(e)
            raise e

    file_paths = []
    for fn in filenames:
        file_path = os.path.join(root, fn)
        file_paths.append(file_path)
    return file_paths


def read_dirs(paths, cache_dir):
    img_filenames = []
    for path in paths:
        file_paths = get_file_paths(path, cache_dir)
        img_filenames.append(file_paths)

    return img_filenames


# --------------------------------------------
# get uint8 image of size HxWxn_channles (RGB)
# --------------------------------------------

def imread_uint(path, n_channels=3, return_orig=False):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        orig_img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(orig_img, axis=2)  # HxWx1
    elif n_channels == 3:
        orig_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        #try:
        if orig_img.ndim == 2:
            img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)  # RGB
        # except AttributeError as e:
        #     print("=========== \n", path, "\n =================")
        #     raise e

    if return_orig:
        return img, orig_img
    else:
        return img


# This method will show image in any image viewer
def imread_PIL(path):
    img = Image.open(path)
    return img


# --------------------------------------------
# numpy(single) [0, 1] <--->  numpy(uint)
# --------------------------------------------


def uint2single(img):
    return np.float32(img/255.)


def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())


# --------------------------------------------
# numpy(single) (HxWxC) <--->  tensor
# --------------------------------------------


# convert single (HxWxC) to 3-dimensional torch tensor
def single2tensor3(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


# --------------------------------------------
# numpy(uint) (HxWxC or HxW) <--->  tensor
# --------------------------------------------


# convert 2/3/4-dimensional torch tensor to uint
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())


# TODO: Insert comment here to describe the functionality
def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# --------------------------------------------
# matlab's imwrite
# --------------------------------------------
def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)
