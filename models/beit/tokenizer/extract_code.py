# --------------------------------------------------------
# BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers (https://arxiv.org/abs/2208.06366)
# Github source: https://github.com/microsoft/unilm/tree/master/beitv2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Zhiliang Peng
# Based on BEiT, timm, DeiT and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

from pathlib import Path
from datetime import datetime
import argparse

from utils.distributed import set_random_seed
from data.kmeans.dataset_dloader_sequence_kmeans import custom_dataloader
import torch
from torchvision import transforms as pth_transforms
from timm.models import create_model

import pandas as pd
import vqkd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_transform(args):
    if args.use_timm_transform:
        from timm.data.transforms import RandomResizedCropAndInterpolation
        transform = pth_transforms.Compose([
            RandomResizedCropAndInterpolation(224, scale=(1.0, 1.0), ratio=(1.0, 1.0), interpolation='bicubic'),
            pth_transforms.ToTensor(),
        ])
    else:
        transform = pth_transforms.Compose([
            pth_transforms.Resize(224, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
        ])

    return transform


def save_codes(codes, bool_val, args, iteration_number=None, prefix=""):
    if iteration_number is None:
        iteration_number = 1000000

    mode = "val" if bool_val else "train"
    codes_save_path = args.codes_save_path / mode / f"{prefix}{args.dataset_name}_codes_codebook-size-{args.codebook_size}_batchindex-{iteration_number + 1}_bsize-{args.batch_size}_{mode}.pth"
    print(f"Saving codes to {codes_save_path}")
    codes_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(codes, codes_save_path)

    saved_codes = torch.load(codes_save_path)

    # check if codes are saved correctly
    assert torch.all(torch.eq(codes, saved_codes)), "[X] Codes not saved correctly"
    print("[V] Codes saved correctly")


def extract_one_epoch(model, dataloader, bool_val, args):
    print("Extracting one epoch")
    i = 0
    save_freq = 500  # must be >1
    num_digits = len(str(len(dataloader) // save_freq))
    save_index = 0
    flag = True
    for i, (images, _) in enumerate(dataloader):
        print(f"Extracting batch {i}/{len(dataloader)}")
        with torch.no_grad():
            images = images.to(device)
            codes = model.get_codebook_indices(images).reshape(images.shape[0], -1)
            assert 0 <= codes.min() <= codes.max() < args.codebook_size, "Codebook indices out of range"
            if flag:
                all_codes = codes
                flag = False
            else:
                all_codes = torch.cat((all_codes, codes), dim=0)

        if (i + 1) % save_freq == 0:
            print(f"Saving codes at batch {i}")
            save_index += 1
            prefix = f"{save_index:0{num_digits}}_"
            save_codes(all_codes, bool_val, args, i, prefix=prefix)
            flag = True

    print(f"Saving codes at batch {i}")
    save_index += 1
    prefix = f"{save_index:0{num_digits}}_"
    save_codes(all_codes, bool_val, args, i, prefix=prefix)


def get_code(args):
    # ============ preparing data ... ============
    transform = get_transform(args)
    print(f"Image transforms: {transform}")

    dataloaders = custom_dataloader(
        args.dataset_csv_train,
        args.dataset_csv_test,
        args.dataset_path,
        args.batch_size,
        transform,
        num_workers=args.num_workers,
    )

    # ============ building network ... ============
    if args.dall_e:
        model = create_d_vae(
            weight_path=args.discrete_vae_weight_path, d_vae_type=args.discrete_vae_type,
            device=device, image_size=args.second_input_size)
    else:
        model = create_model(
            args.model,
            pretrained=True,
            pretrained_weight=args.pretrained_weights,
            as_tokenzer=True,
        )
    model.to(device)
    model.eval()

    args.codes_save_path = Path(
        args.codes_save_path) / f"{args.dataset_name}_{args.model}{'_timm_transform' if args.use_timm_transform else '_resize-112-crop-112'}_seed-{args.seed}"
    args.codes_save_path.mkdir(parents=True, exist_ok=True)

    # ============ extracting codes ... ============
    extract_one_epoch(model, dataloaders['train'], False, args)
    extract_one_epoch(model, dataloaders['val'], True, args)


def verify_codes(codes_path, mode="train", plot_compare=False):
    import matplotlib.pyplot as plt

    transform = get_transform(args)
    print(f"Image transforms: {transform}")

    dataloader = custom_dataloader(
        args.dataset_csv_train,
        args.dataset_csv_test,
        args.dataset_path,
        1,
        transform,
        num_workers=args.num_workers,
    )[mode]
    if args.dall_e:
        model = create_d_vae(
            weight_path=args.discrete_vae_weight_path, d_vae_type=args.discrete_vae_type,
            device=device, image_size=args.second_input_size)
    else:
        model = create_model(
            args.model,
            pretrained=True,
            pretrained_weight=args.pretrained_weights,
            as_tokenzer=True,
        )
    model.to(device)
    model.eval()

    codes_path = Path(codes_path)
    patches = torch.zeros((0, 196), dtype=torch.int16)
    if codes_path.is_dir():
        file_list = os.listdir(codes_path)
        for index, file in enumerate(file_list):
            file_list[index] = os.path.join(codes_path, file)
        file_list = sorted(file_list, key=os.path.getmtime)
        for file in file_list:
            print(os.path.join(codes_path, file))
            t = torch.load(os.path.join(codes_path, file), map_location='cpu')
            patches = torch.cat((patches, t), dim=0)
        img_post_processed = patches
    else:
        img_post_processed = torch.load(codes_path, map_location='cpu')
    # print info about img_post_processed
    print(f"img_post_processed.shape: {img_post_processed.shape}")
    print(f"img_post_processed.dtype: {img_post_processed.dtype}")
    print(f"img_post_processed.min(): {img_post_processed.min()}")
    print(f"img_post_processed.max(): {img_post_processed.max()}")

    mismatch = {}
    # create timestamp in format yyyy-mm-dd_hh-mm-ss
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = f"./{mode}_mismatch_{timestamp}.csv"
    print(f"Saving mismatch info to '{csv_path}'")

    # for every image in dataloader, get the codebook indices
    for i, (images, _) in enumerate(dataloader):

        if (i + 1) % 100 == 0:
            print("Processed {} images".format(i + 1))

        img = images[0].permute(1, 2, 0).numpy()
        saved_code = img_post_processed[i].reshape(14, 14).numpy()

        with torch.no_grad():
            images = images.to(device)
            model_codes = model.get_codebook_indices(images)[0].reshape(14, 14).cpu().numpy()

        if (saved_code != model_codes).any():
            num_mismatch = (saved_code != model_codes).sum()
            mismatch[i] = (num_mismatch, saved_code[saved_code != model_codes], model_codes[saved_code != model_codes])
            print(f"Image {i} has {num_mismatch} mismatched codes: {mismatch[i]}")
            print(f"Average number of mismatch per image: {np.mean([mismatch[i][0] for i in mismatch])}")

            # save mismatch as csv
            mismatch_df = pd.DataFrame.from_dict(
                mismatch,
                orient='index',
                columns=['num_mismatch', 'saved_code', 'model_codes']
            )

            mismatch_df.to_csv(csv_path)

        if plot_compare:
            # plot code over img and plot the result
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(img)
            ax[1].imshow(model_codes)
            ax[2].imshow(saved_code)

            # add color bar to plots
            for j in range(3):
                # hide x-axis and y-axis
                ax[j].axis('off')

            plt.tight_layout()
            plt.show()
            if i == 10:
                break

    print(f"mismatch: {mismatch}")
    print(f"Total number of mismatch: {np.sum([mismatch[i][0] for i in mismatch])}")
    print(f"Average number of mismatch per image: {np.mean([mismatch[i][0] for i in mismatch])}")


def check_csv_content(path_1, path_2):
    # load csv files
    df1 = pd.read_csv(path_1, index_col=0)
    df2 = pd.read_csv(path_2, index_col=0)

    # make df1 and df2 have the number of rows
    if df1.shape[0] > df2.shape[0]:
        df1 = df1.iloc[:df2.shape[0], :]
    elif df1.shape[0] < df2.shape[0]:
        df2 = df2.iloc[:df1.shape[0], :]

    df1 = df1.reset_index()
    df2 = df2.reset_index()

    # check df1 and df2 are identical
    assert (df1 == df2).all().all(), "CSV files are not identical"
    print(f"CSV shape: {df1.shape[0]}")
    print(f"Total mismatch: {df1.iloc[:, 0].sum()}")
    print("CSV files are identical")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get code for VQ-KD')
    parser.add_argument(
        '--model',
        default='vqkd_encoder_base_decoder_3x768x12_clip',
        type=str,
        help="model"
    )
    parser.add_argument(
        '--pretrained_weights',
        default='pretrained/beit_v2/vqkd_encoder_base_decoder_3x768x12_clip-d5036aa7.pth',
        type=str,
        help="Path to pretrained weights to evaluate."
    )
    parser.add_argument(
        '--dataset_path',
        default='PATH/TO/DATASET',
        type=str,
        help="dataset path"
    )
    parser.add_argument(
        '--codes_save_path',
        default="PATH/TO/CODEBOOKS",
        type=str,
        help='file path to save the extracted codes'
    )
    parser.add_argument(
        '--codebook_size',
        default=8192,
        type=int,
        help='number of visual tokens'
    )

    parser.add_argument('--dall_e', action='store_true', default=False, help='use DALL-E')
    parser.add_argument("--discrete_vae_weight_path", type=str, default="/PATH/TO/codebooks/dall_e_tokenizer_weight")
    parser.add_argument("--discrete_vae_type", type=str, default="dall-e")
    parser.add_argument('--second_input_size', default=112, type=int,
                        help='images input size for discrete vae')
    parser.add_argument('--dataset_name', default='imagenet', type=str)

    parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--num_workers', default=4, type=int, help='number of processes loading data')
    parser.add_argument('--dataset_csv_train', type=str)
    parser.add_argument('--dataset_csv_test', type=str)
    parser.add_argument('--use_timm_transform', action='store_true')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

    args = parser.parse_args()

    set_random_seed(args.seed, deterministic=True)
    print(f"Random seed: {args.seed}")

    # === Extract codebook indices ===
    get_code(args)

    # === Check codebook indices ===
    plot_compare = False
    codes_path = "/codebooks/beit_v2_vqkd_encoder_base_decoder_3x768x12_clip_resize-224-crop-224_seed-42/val"
    print(f"\n\n=== Verifying codes at path: '{codes_path}' ===")
    verify_codes(codes_path, mode="val", plot_compare=plot_compare)
    codes_path = "/codebooks/beit_v2_vqkd_encoder_base_decoder_3x768x12_clip_resize-224-crop-224_seed-42/train"
    print(f"\n\n=== Verifying codes at path: '{codes_path}' ===")
    verify_codes(codes_path, mode="train", plot_compare=plot_compare)
