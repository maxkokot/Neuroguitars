import os
import re
import torch
from torchvision.utils import make_grid

from matplotlib import pyplot as plt


def make_samples(model, num_preds=16):

    z = torch.randn(num_preds, model.latent_dim)

    with torch.no_grad():
        pred = model(z)

    pred = pred * 0.5 + 0.5
    grid = make_grid(pred).permute(1, 2, 0).numpy()

    return grid


def save_im(grid, pic_dir):

    p = re.compile(r'img_\d*[.]jpg')
    list_imgs = list(filter(lambda pth:  p.match(pth),
                            os.listdir(pic_dir)))

    nums = list(map(lambda pth: int(pth.split('img_')[-1]
                                    .split('.')[0]),
                    list_imgs))

    if nums:
        num = max(nums) + 1
    else:
        num = 0

    pic_name = 'img_{}.jpg'.format(num)
    path = os.path.join(pic_dir, pic_name)

    fig = plt.figure(figsize=(8, 3), dpi=300)
    plt.imshow(grid)
    plt.savefig(path)
    plt.close(fig)
