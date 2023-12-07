from pathlib import Path

import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms as T


class DreamBoothDataset(Dataset):
    def __init__(self, images_dir: str, prompt: str, size: int = 1024):
        self.size = size
        self.images_dir = Path(images_dir)
        if not self.images_dir.exists():
            raise ValueError("images dir does not exist")

        self.prompt = prompt

        self.instances_images = [Image.open(path) for path in self.images_dir.iterdir()]
        self.num_instance_images = len(self.instances_images)

        self.image_transforms = T.Compose(
            [
                T.Resize(self.size, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(self.size),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self.num_instance_images

    def __getitem__(self, index):
        example = {}

        instance_image = self.instances_images[index % self.num_instance_images]
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["instance_images"] = self.image_transforms(instance_image)
        example[
            "instance_prompt"
        ] = self.prompt  # all images will have this same prompt

        return example


def collate_fn(examples):
    pixel_values = [e["instance_images"] for e in examples]
    prompts = [e["instance_prompt"] for e in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}

    return batch
