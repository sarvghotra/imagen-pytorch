import os
import torch
from imagen_pytorch import Unet, SRUnet256, Imagen, ImagenTrainer
import argparse
from utils_dist.utils_dist import get_dist_info, init_dist
from data.select_dataset import create_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist
import utils.utils_image as util
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1384053, help='seed for reproducibility')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=True)

    opt = vars(parser.parse_args())
    seed = opt['seed']

    # Setting it here for debugging reasons
    # FIXME: remove it
    dataset_opt = {
        'dataloader_batch_size': 128,
        'dataloader_num_workers': 192,
        'dataset_type': 'imgtext',
        'images_data': ['/sr_img_data/users/saghotra/datasets/img/train/mscoco_train2017/train2017/'],
        'annotations_file': '/sr_img_data/users/saghotra/datasets/img/train/mscoco_train2017/annotations/captions_train2017.json',
        'text_embed_dir': '/data/users/saghotra/models/img_gen/imagen/small/train_emb/',
        'text_embed_msk_dir': '/data/users/saghotra/models/img_gen/imagen/small/train_emb_msk/',
        'cache_dir': '/sr_img_data/users/saghotra/datasets/img/cache/',
        'image_size': 64
    }

    val_dataset_opt = {
        'dataloader_batch_size': 4,
        'dataloader_num_workers': 64,
        'dataset_type': 'imgtext',
        'images_data': ['/sr_img_data/users/saghotra/datasets/img/train/mscoco_train2017/val2017_try2/val2017/'],
        'annotations_file': '/sr_img_data/users/saghotra/datasets/img/train/mscoco_train2017/annotations/captions_val2017.json',
        'text_embed_dir': '/data/users/saghotra/models/img_gen/imagen/small/val_emb/',
        'text_embed_msk_dir': '/data/users/saghotra/models/img_gen/imagen/small/val_emb_msk/',
        'cache_dir': '/sr_img_data/users/saghotra/datasets/img/cache/',
        'image_size': 64
    }

    task_name = "tmp" # FIXME
    epochs = 100
    start_epoch = 2
    current_step = 9999
    val_size = 1 # 15 FIXME
    val_freq = 2    # 10000 FIXME
    log_freq = 10
    val_output_dir = '/data/users/saghotra/models/img_gen/imagen/small/{}/val_output_imgs'.format(task_name)
    ckpt_save_path = '/data/users/saghotra/models/img_gen/imagen/small/{}/models'.format(task_name)
    tb_path = os.path.join('/data/users/saghotra/models/img_gen/imagen/tb_logs', task_name)
    opt['dataset_opt'] = dataset_opt
    opt['val_dataset_opt'] = val_dataset_opt

    pretrained_ckpt = '/data/users/saghotra/models/img_gen/imagen/small/mscocotrain2017_e1_try2/models/2_753/save_obj.pth'


    if opt['dist']:
        init_dist('pytorch')
        opt['rank'], opt['world_size'] = get_dist_info()

    opt['num_gpu'] = 0
    if dist.is_available() and dist.is_initialized():
        opt['num_gpu'] = opt['world_size']
    else:
        opt['num_gpu'] = torch.cuda.device_count()


    # model hyper-params
    lowres_cond = False

    # unet for imagen
    unet1 = Unet(
        attn_dim_head= 64,
        attn_heads= 8,
        attn_pool_text= True,
        cond_on_text= True,
        dim_mults= [1, 2, 3, 4],
        dim=192,
        ff_mult= 1.0,
        init_conv_to_final_conv_residual= True,
        layer_attns= [False, True, True, True],
        layer_cross_attns= [False, True, True, True],
        lowres_cond= lowres_cond,
        use_linear_attn= False,
        cond_dim = 512,
        num_resnet_blocks=2,
        # memory_efficient= True,
        )

    if opt['rank'] == 0:
        print("No of parameters: ", sum(p.numel() for p in unet1.parameters() if p.requires_grad))
        tb_writer = SummaryWriter(tb_path)
        # print(unet1)

    # FIXME: remove it
    opt['dataset_opt']['rank'] = opt['rank']
    opt['val_dataset_opt']['rank'] = opt['rank']

    # imagen, which contains the unets above (base unet and super resoluting ones)
    imagen = Imagen(
        unets = (unet1,),
        text_encoder_name = 't5-large',
        image_sizes = (64,),
        timesteps = 1000,
        cond_drop_prob = 0.1,
        p2_loss_weight_gamma = 1.0,
        lowres_cond = lowres_cond,
    )   # .cuda() FIXME?

    #############################
    #####  Create dataset #######
    #############################

    train_set = create_dataset(opt['dataset_opt'])
    val_set = create_dataset(opt['val_dataset_opt'])

    if opt['dist']:
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True, seed=seed)
        shuffle = False
    else:
        train_sampler = RandomSampler(train_set, drop_last=True, seed=seed)
        shuffle = True

    train_loader = DataLoader(train_set,
                                batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                shuffle=shuffle,
                                num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                drop_last=True,
                                pin_memory=True,
                                sampler=train_sampler)

    val_loader = DataLoader(val_set,
                                batch_size=val_dataset_opt['dataloader_batch_size'],
                                shuffle=False,
                                num_workers=val_dataset_opt['dataloader_num_workers'],
                                drop_last=False,
                                pin_memory=True)

    # wrap imagen with the trainer class
    trainer = ImagenTrainer(imagen,
                                warmup_steps=2000,
                                cosine_decay_max_steps=len(train_loader) * epochs)

    # load ckpt and other save states
    if pretrained_ckpt is not None:
        trainer.load(pretrained_ckpt)

    trainer.move_to_device()

    avg_loss = []
    for ep in range(start_epoch, epochs):
        if opt['rank'] == 0:
            print("Starting epoch: ", ep)
        if opt['dist']:
            train_sampler.set_epoch(ep)

        for i, train_data in enumerate(train_loader):
            current_step += 1

            images = train_data['images'].to(imagen.device)
            text_embeds = train_data['text_embeds'].to(imagen.device)
            text_masks = train_data['text_masks'].to(imagen.device)

            loss = trainer(
                images,
                text_embeds = text_embeds,
                text_masks = text_masks,
                unet_number = 1,        # harcoded for now, # FIXME
            )

            avg_loss.append(loss)

            #max_batch_size = opt['batch_size']        # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory

            trainer.update(unet_number = 1)

            if opt['rank'] == 0:
                tb_writer.add_scalar('LR', trainer.get_lr(0), current_step)

                if current_step % log_freq == 0 and current_step > 0:
                    if len(avg_loss) > 0:
                        print("step: ", current_step, " loss: ", sum(avg_loss)/len(avg_loss))
                        tb_writer.add_scalar('Loss/train', sum(avg_loss)/len(avg_loss), current_step)
                    avg_loss = []

            if opt['rank'] == 0 and current_step % val_freq == 0 and current_step > 0:
                # save checkpoint
                save_path = os.path.join(ckpt_save_path, str(current_step), "save_obj.pth")
                print(save_path)
                if not os.path.exists(save_path):
                    trainer.save(save_path)
                    print("saved")
                else:
                    print("{} checkpoint already exists!!!".format(save_path))

                with torch.no_grad():
                    for val_data, _ in zip(val_loader, range(val_size)):
                        text_embeds = val_data['text_embeds'].to(imagen.device)
                        text_masks = val_data['text_masks'].to(imagen.device)
                        img_names = val_data['img_names']
                        captions = val_data['captions']
                        # texts = [
                        #     'a whale breaching from afar',
                        #     'young girl blowing out candles on her birthday cake',
                        #     'fireworks with blue and green sparkles'
                        # ]
                        val_images = imagen.sample(text_embeds = text_embeds, text_masks = text_masks, cond_scale = 5.)

                        # print(images.shape) # (3, 3, 256, 256)
                        for val_img, img_name, caption in zip(val_images, img_names, captions):
                            print("saved")
                            val_img_out = util.tensor2uint(val_img)

                            img_name = '.'.join(img_name.split('.')[:-1])
                            val_img_output_dir = os.path.join(val_output_dir, img_name)
                            print(val_img_output_dir)
                            if not os.path.exists(val_img_output_dir):
                                os.makedirs(val_img_output_dir)

                            capt_img_name = "_".join(caption.split(' '))
                            print("capt_img_name: ", capt_img_name)
                            util.imsave(val_img_out, os.path.join(val_img_output_dir, capt_img_name + "_{}.png".format(current_step)))



if __name__ == "__main__":
    main()

    # TODO:
    #   [Done] 2. write dataset code
    #       - verify it (see output)

    #   [Done] 3. val dataset

    #   4. compute metrics?
    #   [Done] 5. save samples while training

    #   [Done] 6. save checkpoints

    #   [Done] 1. change hyper-params to actual values

    #   plot LR and loss curves