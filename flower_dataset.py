import os.path as osp
import torch

from torch.utils.data import Dataset
from torchvision import transforms

from glob import glob
from PIL import Image


class DreamBoothDataset(Dataset):
    def __init__(self, dataset_dir, instance_prompt, tokenizer, size=512):

        self.dataset = glob(osp.join(dataset_dir, '*.jpg'))
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.size = size
        self.transforms = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = {}
        image = Image.open(self.dataset[index]).convert('RGB')
        example["instance_images"] = self.transforms(image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        return example