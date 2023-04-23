import math
import torch
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, PNDMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from flower_dataset import DreamBoothDataset
from transformers import CLIPTokenizer

from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel

from argparse import Namespace

from utils import image_grid


if __name__ == '__main__':

    # Inference!!
    pipe = StableDiffusionPipeline.from_pretrained(
        "my-dreambooth",
        torch_dtype=torch.float16,
    ).to("cuda")

    prompt = input("Please put prompt")
    print(f'prompt : {prompt}')

    # Tune the guidance to control how closely the generations follow the prompt.
    # Values between 7-11 usually work best
    guidance_scale = 7

    num_cols = 2
    all_images = []
    for _ in range(num_cols):
        images = pipe(prompt, guidance_scale=guidance_scale).images
        all_images.extend(images)

    result_image = image_grid(all_images, 1, num_cols)
    result_image.save(f'{prompt}.jpg')