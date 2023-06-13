import os
import sys

sys.path.append(os.getcwd())
import argparse
import time

import torch

import clip
from dataset_dloader_sequence_kmeans import custom_dataloader, post_processing_images, PatchIndexExtractor

#  TINY: 192 dim internal, 3 multihead e 12 layer
#  SMALL: 384 dim internal, 6 heads e 12 layer
#  BASE: 782 dim internal, 12 heads, 6 layer


######################################################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description='VITransformer training script')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--num_workers', default=6, type=int, help='number of processes loading data')
    parser.add_argument('--num_token', default=8192, type=int, help='number of possible index outputted by kmeans')
    parser.add_argument(
        '--checkpoint_path', type=str,
        help='file path to save the model'
    )
    parser.add_argument(
        '--kmeans_folder', type=str,
        help='folder of npy checkpoint of faiss kmeans'
    )
    parser.add_argument('--dataset_csv_train', type=str)
    parser.add_argument('--dataset_csv_test', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--max_patches', type=int)
    parser.add_argument('--image_model', type=str, help='clip path of architecture weight')
    args = parser.parse_args()
    return args


def main(args):
    print("Inside main")

    print("Device", device)
    _, transform = clip.load(args.image_model, jit=False)
    dataloaders = custom_dataloader(
        args.dataset_csv_train,
        args.dataset_csv_test,
        args.dataset_path,
        args.batch_size,
        transform,
        num_workers=args.num_workers,
    )
    patch_ext = PatchIndexExtractor(
        kmeans_folder=args.kmeans_folder,
        n_centroids=args.num_token, n_iter=20, max_patches=args.max_patches,
        embedding_dim=4096
    )

    extract_one_epoch(dataloaders['train'], patch_ext, False, args)
    extract_one_epoch(dataloaders['val'], patch_ext, True, args)


def save_patches(patches, bool_val, args, iteration_number=None):
    if iteration_number is None:
        iteration_number = 1000000
    if bool_val:
        torch.save(patches, args.checkpoint_path + '/k_means_processed_ade_' + str(args.num_token) + '_val.pth')
    else:
        torch.save(
            patches,
            args.checkpoint_path + '/ade_' + str(args.num_token) + '/k_means_processed_ade' + str(
                args.num_token) + '_index_%d_bsize_%d.pth' % (
                iteration_number, args.batch_size)
        )


def extract_one_epoch(train_loader, patch_ext, bool_val, args):
    patches = torch.zeros((0, 196), dtype=torch.int16)
    clip_model, _ = clip.load(args.image_model, jit=False)
    image_model = clip_model.visual
    image_model.to(device)
    image_model.eval()
    for index, (inputs, idx) in enumerate(train_loader):
        if index % 100 == 0:
            print("\t Batch:", index, "; inputs shape:", inputs.shape)
        start = time.time()
        with torch.no_grad():
            features = image_model.intermediate_features(
                inputs.to(device)
            ).cpu().float().numpy()
        end = time.time()
        print("Processed, clip_images in: ", end - start)
        start = end
        inputs = post_processing_images(patch_ext, features)
        end = time.time()
        print("Processed, kmeans in: ", end - start)

        patches = torch.cat((patches, inputs), dim=0)
        # gImportant if dataset is big
        if index % 500 == 0 and not bool_val:
            print("saving...")
            save_patches(patches, bool_val, args, index)
            del patches
            patches = torch.zeros((0, 196), dtype=torch.int16)
    save_patches(patches, bool_val, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
