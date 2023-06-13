import logging
import os
import pandas as pd
import time
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from utils.models import attention_mask_full_mapet_1

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

_logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    def __init__(
            self,
            patches_folder,
            annotations_file,
            sequence_len=None,
            num_attention_heads=None,
            least_number=1,
            isDir=False,
            pre_trained=False,
            transform=None,
            positional=False,
            root_path="",
            mapet=False,
    ):
        """
        Custom dataset for images and tokens

        """
        super().__init__()

        self.transform = transform
        self.p_folder = patches_folder
        self.img_labels = pd.read_csv(annotations_file)
        patches = torch.zeros((0, 196), dtype=torch.int16)
        self.positional = positional
        if isDir:
            file_list = os.listdir(self.p_folder)
            for index, file in enumerate(file_list):
                file_list[index] = os.path.join(self.p_folder, file)
            file_list = sorted(file_list, key=os.path.getmtime)
            for file in file_list:
                _logger.info(os.path.join(self.p_folder, file))
                t = torch.load(os.path.join(self.p_folder, file), map_location='cpu')
                patches = torch.cat((patches, t), dim=0)
            self.img_post_processed = patches
        else:
            self.img_post_processed = torch.load(self.p_folder, map_location='cpu')
            _logger.info(self.p_folder)
        self.sequence_len = sequence_len
        self.num_attention_heads = num_attention_heads
        self.least_number = least_number
        self.pre_trained = pre_trained
        self.path_root = root_path
        self.mapet = mapet
        logging.info(f"Created Dataset with visual tokens on {len(self)} images")

    def __getitem__(self, idx):
        img_path = os.path.join(self.path_root, self.img_labels.iloc[idx, 0])
        target = self.img_labels.iloc[idx, 1]
        image = default_loader(img_path)
        if self.transform is not None:
            image = self.transform(image)
        kmeans_indices = self.img_post_processed[idx, :]
        mask_query, mask_content, index_vector, upper_predictions = self.get_permutation()
        return image, target, kmeans_indices, mask_query, mask_content, index_vector, upper_predictions

    def __len__(self):
        return self.img_post_processed.shape[0]

    def get_permutation(self):
        # 1  is the index of classification token
        permutation = torch.cat([torch.ones(1), torch.randperm(self.sequence_len - 1) + 2], dim=0)
        mask_query, mask_content, index_vector, upper_predictions = attention_mask_full_mapet_1(
            permutation,
            self.num_attention_heads,
            self.least_number,
            197
        )
        return mask_query, mask_content, index_vector, upper_predictions


def post_processing_images(clip_index, index_vector, fake_token=8193):
    '''
    Function that apply kmeans to a batch of images, then extract labels of the element to predict (knowing that only least element of query vector are used)

    :param index_vector: Batched tensor in wich there are the indexes of elements with least_element before of them
    :return: Tuple with (batched indexes of the sequence, releated labels)
    '''
    sequence_added = torch.cat((torch.tensor(fake_token).repeat(clip_index.shape[0], 1), clip_index), dim=1)
    sequence_added[index_vector] = fake_token
    return clip_index.int(), sequence_added
