import argparse
import os
import torch
from torchmetrics.image.kid import KernelInceptionDistance
from PIL import Image
import numpy as np
from einops import rearrange

parser = argparse.ArgumentParser()
parser.add_argument("--real", type=str, default='./results/facadesAB/real', help="Path to the real images")
parser.add_argument("--fake", type=str, default='./results/facadesAB/fake', help="Path to the fake images")
parser.add_argument("--gpu_id", type=int, default='0', help="GPU id to use, e.g. 0")
args = parser.parse_args()

sorting_key = lambda f: int(f.split('_')[0]) if len(f.split('_')) > 0 else int(f.split('.')[0])
reals = sorted(os.listdir(args.real), key=sorting_key)
fakes = sorted(os.listdir(args.fake), key=sorting_key) 
device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
kid = KernelInceptionDistance(subset_size=50).to(device)

for i, (real, fake) in enumerate(zip(reals, fakes)):
    print(f'Processing real/fake {i}')
    real_img = torch.tensor(np.array(Image.open(f'{args.real}/{real}').convert('RGB')), dtype=torch.uint8).to(device)
    real_img = rearrange(real_img, 'h w c -> 1 c h w')
    fake_img = torch.tensor(np.array(Image.open(f'{args.fake}/{fake}').convert('RGB')), dtype=torch.uint8).to(device)
    fake_img = rearrange(fake_img, 'h w c -> 1 c h w')
    kid.update(real_img, real=True)
    kid.update(fake_img, real=False)

kid_mean, kid_std = kid.compute()
print(f'KID (scaled by 100): {100*kid_mean:.2f} +/- {100*kid_std:.2f}')