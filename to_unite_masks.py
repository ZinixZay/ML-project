import torch

from torchvision.transforms.v2 import PILToTensor, ToPILImage
from PIL import Image
import matplotlib.pyplot as plt

import os

tr = PILToTensor()
tr_back = ToPILImage()

i1 = tr(Image.open('data/train_lung_masks/img_0.png'))
i2 = tr(Image.open('data/train_images/img_0.png'))

i2 = (i1 > 100).int() * i2

plt.imshow(get_borders(i2))
plt.show()

