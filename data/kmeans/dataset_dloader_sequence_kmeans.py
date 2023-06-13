import faiss
import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn as nn
import torchvision.transforms as transform
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, DistributedSampler, SequentialSampler, Subset
from torchvision.datasets.folder import default_loader

import data.kmeans.clip as clip

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PatchIndexExtractor():
    def __init__(self, kmeans_folder: str, n_centroids, n_iter, max_patches, embedding_dim=4096, max_points_per_centroid=512):
        """
        Class to process images and extract sequence indexes with kmeans

        :param kmeans_folder: folder in which kmeans centroids and indexes are saved
        :param n_centroids: number of centroids of kmeans algorithm
        :param n_iter: number of iteration in kmeans
        :param max_patches: number of patches
        """
        self.kmeans = faiss.Kmeans(
            embedding_dim, n_centroids, niter=n_iter, verbose=True, nredo=5, gpu=False,
            max_points_per_centroid=max_points_per_centroid
        )
        self.kmeans.centroids = np.load(
            kmeans_folder + "/codebook_centroids_%d_%d.npy" % (max_patches, n_centroids)
        )
        self.kmeans.index = faiss.read_index(
            kmeans_folder + "/codebook_index_%d_%d.npy" % (max_patches, n_centroids)
        )

    def patch_extractor(self, features):
        '''
        Function to extract sequence indexes for batch of data

        :param image: the image to process and extract index_value
        :param raw_pixel: boolean to avoid the application of self.image_model and the extraction of features
        :return:  indexes of patch processed by kmeans
        '''
        batch_size = features.shape[0]
        features = features.reshape(-1, features.shape[-1])
        start = time.time()
        _, codes = self.kmeans.index.search(np.ascontiguousarray(features), 1)
        end = time.time()
        print("Index search in: ", end - start)
        codes = codes.reshape(batch_size, -1)
        codes = torch.from_numpy(codes)
        return codes


class SequenceDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transforms=None, target_transforms=None):
        """
        Custom dataset for images. This dataset for

        :param annotations_file: csv file for labelled data
        :param img_dir: Image directory
        :param kmeans: kmeans to process element
        :param transforms: input transforms
        :param target_transforms: target transforms

        """
        super().__init__()
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transforms
        self.target_transform = target_transforms

    def __getitem__(self, idx):
        # get img_path from csv
        img_path = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, str(img_path))
        image = default_loader(img_path)
        target = self.img_labels.iloc[idx, 1]
        # perform transformation on the dataset
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.img_labels.index)


def custom_dataloader(
        csv_path_train,
        csv_path_test,
        image_directiory,
        batch_size,
        t=None,
        num_workers=4,
):
    if t is None:
        t = transform.Compose(
            [transform.ToTensor(), transform.Resize((224, 224), interpolation=BICUBIC),
             transform.Normalize(
                 (0.48145466, 0.4578275, 0.40821073),
                 (0.26862954, 0.26130258, 0.27577711)
             )]
        )
    d_train = SequenceDataset(
        csv_path_train, image_directiory, transforms=t
    )
    d_val = SequenceDataset(
        csv_path_test, image_directiory, transforms=t
    )

    dataset_size = len(d_train)
    print("Train Dataset size:", dataset_size)
    dataset_size = len(d_val)
    print("Eval Dataset size:", dataset_size)

    train_sampler = SequentialSampler(d_train)
    valid_sampler = SequentialSampler(d_val)

    train_loader = DataLoader(d_train, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
                              pin_memory=True)
    validation_loader = DataLoader(d_val, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers,
                                   pin_memory=True)

    return {'train': train_loader, 'val': validation_loader}


def post_processing_images(kmeans: PatchIndexExtractor, batch_images):
    '''
    Function that apply kmeans to a batch of images, then extract labels of the element to predict (knowing that only least element of query vector are used)

    :param kmeans: PatchIndexExtractor instance (see class)
    :param batch_images: the batch of images to exctract index
    :param index_vector: Batched tensor in wich there are the indexes of elements with least_element before of them
    :return: Tuple with (batched indexes of the sequence, releated labels)
    '''
    sequence = kmeans.patch_extractor(batch_images)
    return sequence.int()


if __name__ == "__main__":
    KMEANS_FOLDER = Path("/KMEANS/centroids_3000000/")
    IMAGENET_LABELS_TRAIN = Path("./Imagenet_labels.csv")
    IMAGENET_LABELS_EVAL = Path("./Imagenet_labels_test.csv")

    num_heads = 12
    sequence_len = 197
    clip_model, transform = clip.load("clip_RN50x64"[5:], jit=False)
    params = sum(p.numel() for p in clip_model.parameters())
    print(f"#Parameters CLIP: {params}")
    image_model = clip_model.visual
    params = sum(p.numel() for p in image_model.parameters())
    print(f"#Parameters CLIP Visual: {params}")
    image_model.to(device)
    image_model.eval()
    dataloaders = custom_dataloader(
        IMAGENET_LABELS_TRAIN,
        IMAGENET_LABELS_EVAL,
        "",
        1,
        transform,
        num_workers=0,
    )

    patch_ext = PatchIndexExtractor(
        kmeans_folder=str(KMEANS_FOLDER),
        n_centroids=8192,
        n_iter=20,
        max_patches=1000000,
        embedding_dim=4096,
    )
    for epoch in range(1):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # Iterate over data
            for index, (inputs, masks_query, mask_content, index_vector, idx) in enumerate(dataloaders[phase]):
                if index % 60 == 0:
                    print("\t Batch:", index, "; inputs shape:", inputs.shape)
                sequence, labels = post_processing_images(patch_ext, inputs, index_vector)
                print(sequence)
            break
