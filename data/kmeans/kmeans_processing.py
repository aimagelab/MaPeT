import os
import sys

sys.path.append(os.getcwd())

import argparse
import logging
import random

import faiss
import numpy as np
import torch
import clip
import time
from dataset_dloader_vit_Image_net import custom_dataloader
from pathlib import Path

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
_logger = logging.getLogger('train')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def collect_patches(args):
    # Image model
    print("Loading CLIP model...")
    clip_model, transform = clip.load(args.image_model, jit=False)
    image_model = clip_model.visual
    image_model.to(device)
    image_model.eval()
    print("Done.")
    args.image_dim = image_model.embed_dim  # 4096

    dataloader = custom_dataloader(
        csv_path=args.dataset_csv,
        image_directiory=args.dataset_path,
        batch_size=args.batch_size,
        t=transform,
    )
    dataloader_train = dataloader['train']

    imagenet_size = dataloader_train.dataset.__len__()
    num_imagenet_patches = imagenet_size * 196
    args.dropout = round(args.max_patches / num_imagenet_patches, 3)

    # Start data collection
    num_patches = 0
    patches = torch.zeros((0, args.image_dim), dtype=torch.float16)
    it = 0

    for it, data in enumerate(dataloader_train):
        images, _ = data
        images = images.to(device)
        print("Real image shape", images.shape)
        with torch.no_grad():
            images = image_model.intermediate_features(images).detach().cpu()
        print("CLIP processed images", images.shape)
        this_patches = images.reshape(-1, images.shape[-1])
        n_kept_patches = int(this_patches.shape[0] * args.dropout)
        this_patches = this_patches[torch.randperm(this_patches.shape[0])][:n_kept_patches]
        num_patches += this_patches.shape[0]
        patches = torch.cat((patches, this_patches), dim=0)
        print("Iteration %d - collected %d patches so far" % (it, num_patches))
        if it % args.batch_save_freq == 0:
            torch.save(
                patches,
                args.features_output_folder + f'/codebook_patches{args.max_patches}_it{it}.pth'
            )
            del patches
            patches = torch.zeros((0, args.image_dim), dtype=torch.float16)

    print("Collected enough patches. Stopping.")
    torch.save(
        patches,
        args.features_output_folder + f'/codebook_patches{args.max_patches}_it{it}.pth'
    )

    print("Serialized patches to disk.")
    return None


def perform_kmeans(args):
    print("Loading dataset...")
    if args.singular_file:
        data = torch.load(args.features_output_folder + '/codebook_patches_%d.pth' % (args.max_patches)).float().numpy()
    else:
        data = torch.zeros((0, 4096), dtype=torch.float16)
        file_list = os.listdir(args.features_output_folder)
        for index, file in enumerate(file_list):
            file_list[index] = os.path.join(args.features_output_folder, file)
        file_list = sorted(file_list, key=os.path.getmtime)
        for file in file_list:
            print(os.path.join(args.features_output_folder, file))
            t = torch.load(os.path.join(args.features_output_folder, file), map_location='cpu')
            data = torch.cat((data, t), dim=0)
    print("Done.")
    print("Shape of retrieved data", data.shape)
    data = data.float().numpy()
    niter = 20
    kmeans = faiss.Kmeans(
        data.shape[1],  # input dimension
        args.ncentroids,  # nb of centroids
        niter=niter,
        verbose=True,
        nredo=5,
        gpu=True,
        max_points_per_centroid=args.max_points_per_centroid,
    )
    print("Training kmeans...")
    t1 = time.time()
    kmeans.train(data)
    t2 = time.time()
    print(f"Kmeans training time: {(t2 - t1) / 60} min")
    np.save(
        args.kmeans_output_folder + "/codebook_centroids_%d_%d.npy" % (args.max_patches, args.ncentroids),
        kmeans.centroids
    )
    faiss.write_index(
        faiss.index_gpu_to_cpu(kmeans.index),
        args.kmeans_output_folder + "/codebook_index_%d_%d.npy" % (args.max_patches, args.ncentroids)
    )


if __name__ == '__main__':
    _logger.info('Extracting patches from images...')

    # Argument parsing
    parser = argparse.ArgumentParser(description='CLIP feature extractor')
    parser.add_argument('phase', type=str)
    parser.add_argument('--features_output_folder', type=str)
    parser.add_argument('--kmeans_output_folder', type=str)
    parser.add_argument('--max_patches', type=int)
    parser.add_argument('--ncentroids', type=int, default=8192)
    parser.add_argument('--batch_save_freq', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--max_points_per_centroid', type=int, default=512)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--image_model', type=str, default='clip_RN50x64')
    parser.add_argument('--cineca', action='store_true')
    parser.add_argument('--singular_file', action="store_true")
    parser.add_argument('--dataset_csv', type=str)

    args = parser.parse_args()
    _logger.info(args)

    Path(args.features_output_folder).mkdir(parents=True, exist_ok=True)
    Path(args.kmeans_output_folder).mkdir(parents=True, exist_ok=True)

    args.max_patches = args.max_patches if args.max_patches else args.ncentroids * args.max_points_per_centroid

    if args.cineca:
        args.dataset_path = './'
    else:
        args.dataset_path = './'

    if args.phase == 'collect_patches':
        collect_patches(args)
    elif args.phase == 'perform_kmeans':
        perform_kmeans(args)
