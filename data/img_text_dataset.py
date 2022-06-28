import random
import numpy as np
import torch.utils.data as data
import os
import json
import utils.utils_image as util

import sys

class DatasetImgText(data.Dataset):
    def __init__(self, opts):
        super(DatasetImgText, self).__init__()
        self.n_channels = 3
        self.image_size = opts['image_size']
        # image_paths = util.read_dirs(opts['images_data'], opts['cache_dir'])
        # image_paths = image_paths[0]

        self.img_dir = opts['images_data'][0]
        self.captid_to_imgname, self.captids, self.captids_to_captions = self._get_captionid_to_imgname(opts['annotations_file'])

        self.text_embed_dir = opts['text_embed_dir']
        self.text_embed_msk_dir = opts['text_embed_msk_dir']

    def _get_captionid_to_imgname(self, annot_file):
        in_file = open(annot_file)
        json_data = json.load(in_file)
        image_info = json_data["images"]
        annotations = json_data["annotations"]

        imgid_to_filename = {}
        n_imgs = 0
        for img in image_info:
            imgid_to_filename[str(img["id"])] = img["file_name"]
            n_imgs += 1

        captid_to_imgname = {}
        captids = []
        captids_to_captions = {}
        for i, annot in enumerate(annotations):
            caption_id = str(annot["id"])
            img_id = str(annot["image_id"])
            img_name = imgid_to_filename[str(img_id)]
            captid_to_imgname[caption_id] = img_name
            captids_to_captions[caption_id] = annot["caption"]
            captids.append(caption_id)

        return captid_to_imgname, captids, captids_to_captions


    def __getitem__(self, index):
        # FIXME: Test new changes
        caption_id = self.captids[index]
        caption = self.captids_to_captions[caption_id]
        img_path = os.path.join(self.img_dir, self.captid_to_imgname[caption_id])
        img_name = os.path.basename(img_path)

        #text_embeds = self.text_embeds[img_name]
        # FIXME: remove it
        # caption_id = random.choice(self.imgname_to_captionid[img_name])

        with open(os.path.join(self.text_embed_dir, caption_id + '.npy'), 'rb') as f:
            text_embed = np.load(f) # torch.ones(256, 1024)

        with open(os.path.join(self.text_embed_msk_dir, caption_id + '.npy'), 'rb') as f:
            text_mask = np.load(f) # torch.ones(256).bool()

        img = util.imread_PIL(img_path)
        img = img.convert("RGB")
        # if self.random_crop:
        #     arr = random_crop_arr(pil_image, self.image_size)
        # else:
        img = util.center_crop_arr(img, self.image_size)

        # Copied from guided-diffusion just to remember it
        # arr = img_arr.astype(np.float32) / 127.5 - 1

        H, W, C = img.shape
        img = util.uint2single(img)
        # HWC to CHW, numpy to tensor
        img = util.single2tensor3(img)

        # # save to see the input
        # img_H_out = util.tensor2uint(img)
        # util.imsave(img_H_out, "/tmp/imgs/" + img_name)
        # print("saved")

        return {'text_embeds': text_embed, 'text_masks': text_mask, 'images': img, 'img_names': img_name, 'captions': caption}

    def __len__(self):
        return len(self.captids)


class DatasetText(data.Dataset):
    def __init__(self, opts):
        super(DatasetText, self).__init__()
        in_file = open(opts['caption_file'])
        json_data = json.load(in_file)

        annotations = json_data["annotations"]
        # image_info = json_data["images"]

        self.captions = []
        self.caption_ids = []
        # with open(opts['caption_file'], 'r', encoding='utf-8') as fi:
        for annot in annotations:
            self.captions.append(annot['caption'].strip())
            self.caption_ids.append(annot['id'])
        #self.text_embeds = read_text_embeds(opts['text_embeds'])


    def __getitem__(self, index):
        return {'caption': self.captions[index], 'caption_id': self.caption_ids[index]}

    def __len__(self):
        return len(self.captions)
