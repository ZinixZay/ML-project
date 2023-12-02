import torch
from torchvision.transforms.v2 import PILToTensor, ToPILImage
from PIL import Image
import matplotlib.pyplot as plt

src = PILToTensor()(Image.open('img_0.png'))
mask = PILToTensor()(Image.open('img_0_mask.png'))

binary_mask = (mask > 0.5).int()

lungs = torch.tensor(src * binary_mask, dtype=torch.uint8)

res_image = ToPILImage()(lungs)
plt.imshow(res_image)
plt.show()


