import torch

from torchvision import transforms
from PIL import Image, ImageOps
from src.gan_playground.GAN import GAN
import torch.nn.functional as F
from glob import glob

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fun with GANs")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--tag", type=str, default="gen_imgs", help="Run tag")
    parser.add_argument(
        "--train_count",
        type=int,
        default=1024,
        help="Number of training samples",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size"
    ),
    parser.add_argument(
        "--image_size", type=int, default=64, help="Size of input/generated images nxn"
    )
    parser.add_argument(
        "--latent_size", type=int, default=128, help="Latent size"
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Epoch count"
    )
    parser.add_argument(
        "--lr", type=int, default=0.002, help="Learning rate"
    )
    return parser.parse_args()


args = parse_arguments()

images = []
convert_tensor = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5))]
)
for i, img_path in enumerate(glob(f"{args.dataset}/*")):
    img = Image.open(
        img_path,
    )
    img = img.resize((args.image_size, args.image_size))
    # img = ImageOps.grayscale(img)
    images.append(convert_tensor(img))
    if i > args.train_count:
        break

tensor = torch.stack(images)
print(tensor.min(), tensor.max())


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


g = GAN(int(tensor.shape[2]), int(tensor.shape[1]), args.latent_size)

g.fit(tensor, args.batch_size, get_default_device(), args.epochs, args.lr,args.tag)
