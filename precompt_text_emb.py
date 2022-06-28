import torch
from imagen_pytorch import Unet, SRUnet256, Imagen, ImagenTrainer
import argparse
from utils_dist.utils_dist import get_dist_info, init_dist
from data.select_dataset import create_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from imagen_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME
import torch
import torch.distributed as dist
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1384053, help='seed for reproducibility')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=True)

    opt = vars(parser.parse_args())
    seed = opt['seed']
    epochs = 4

    # Setting it here for debugging reasons
    # FIXME: remove it
    dataset_opt = {
        'tmp': 1,
        'dataloader_batch_size': 256,
        'dataloader_num_workers': 720,
        'dataset_type': 'text',
        'images_data': ['/sr_img_data/users/saghotra/datasets/img/train/mscoco_train2017/train2017/'],
        'caption_file': '/sr_img_data/users/saghotra/datasets/img/train/mscoco_train2017/annotations/captions_val2017.json',
        'cache_dir': '/sr_img_data/users/saghotra/datasets/img/cache/',
        'image_size': 64
    }
    datatype = "val"

    opt['dataset_opt'] = dataset_opt

    if opt['dist']:
        init_dist('pytorch')
        opt['rank'], opt['world_size'] = get_dist_info()

    opt['num_gpu'] = 0
    if dist.is_available() and dist.is_initialized():
        opt['num_gpu'] = opt['world_size']
    else:
        opt['num_gpu'] = torch.cuda.device_count()


    # unet for imagen
    unet1 = Unet(
        dim = 32,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 3,
        layer_attns = (False, True, True, True),
    )

    # imagen, which contains the unets above (base unet and super resoluting ones)
    imagen = Imagen(
        unets = (unet1,),
        text_encoder_name = 't5-large',
        image_sizes = (64,),
        beta_schedules = ('cosine',),
        timesteps = 1000,
        cond_drop_prob = 0.1
    ).cuda()

    # wrap imagen with the trainer class
    trainer = ImagenTrainer(imagen)

    #############################
    #####  Create dataset #######
    #############################

    train_set = create_dataset(opt['dataset_opt'])

    if opt['dist']:
        train_sampler = DistributedSampler(train_set, shuffle=False, drop_last=False, seed=seed)
        shuffle = False
    else:
        train_sampler = RandomSampler(train_set, drop_last=False, seed=seed)
        shuffle = True

    train_loader = DataLoader(train_set,
                                batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                shuffle=shuffle,
                                num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                drop_last=False,
                                pin_memory=True,
                                sampler=train_sampler)

    # texts = [
    # 'a child screaming at finding a worm within a half-eaten apple vhf nhf nh nhf hgnr some hole are you',
    # 'lizard running across the desert on two feet',
    # 'waking up to a psychedelic landscape',
    # 'seashells sparkling in the shallow waters'
    # ]

    # for ep in range(epochs):
    #     print("Starting epoch: ", ep)
    #     if opt['dist']:
    #         train_sampler.set_epoch(ep)

    #emb_store = dist.FileStore("/tmp/emd_store", opt['world_size'])
    #emb_msk_store = dist.FileStore("/tmp/emd_msk_store", opt['world_size'])

    for i, train_data in enumerate(train_loader):

        if i % 100 == 0 and i > 0 and opt['rank'] == 0:
            print(i, " done")

        indices = train_data['caption_id']
        texts = train_data['caption']
        text_embeds, text_masks = t5_encode_text(texts, name = 't5-large', padding='max_length')

        for k, index in enumerate(indices):
            # save npz
            with open('/tmp/{}_emb/{}.npy'.format(datatype, index), 'wb') as f:
                np.save(f, text_embeds[k].cpu().numpy())

            with open('/tmp/{}_emb_msk/{}.npy'.format(datatype, index), 'wb') as f:
                np.save(f, text_masks[k].cpu().numpy())


if __name__ == "__main__":
    main()

    # TODO:
    #   2. write dataset code
    #       - verify it (see output)
    #   3. val dataset
    #   4. compute metrics?
    #   5. save samples while training
    #   6. save checkpoints
    #   1. change hyper-params to actual values
