from PIL import Image
from torchvision.transforms import v2
import numpy as np


src = Image.open('img_0.png')
mask = Image.open('img_0_mask.png')


def contrast_augmentation(src_image, reduce_contrast: bool = False, output_k: int = 3) -> list:
    """
    Generates images with different contrast factor
    :param src_image: PIL.Image type image
    :param reduce_contrast: If func has to generate images with contrast less than original (False must be better)
    :param output_k: How much photos must be generated
    :return: List of PIL.Image variables with different contrasts
    """
    result = list()
    if reduce_contrast:
        contrast_factors = np.linspace(0.1, 2, output_k)
    else:
        contrast_factors = np.linspace(1, 2, output_k)
    for contrast_factor in contrast_factors:
        result.append(v2.functional.adjust_contrast(src_image, contrast_factor))
    return result


def rotation_augmentation(src_image: Image, torch_rotation: bool = False, output_k: int = 3) -> list:
    """
    Generates images with different angle rotation
    :param src_image: PIL.Image type image
    :param torch_rotation: Which method of rotating to use, PIL method as default (torch method is not done yet)
    :param output_k: How much photos must be generated
    :return: List of PIL.Image / torch (depends on method) variables with different rotation
    """
    result = list()
    if torch_rotation:
        pass
    else:
        angles = np.linspace(-5, 5, output_k)
        for angle in angles:
            result.append(src_image.rotate(angle))
    return result


# Use-cases (rotating)
different_rotations_of_src = rotation_augmentation(src)
different_rotations_of_mask = rotation_augmentation(mask)

# use similar args to rotate doubles
different_rotations_of_src_2 = rotation_augmentation(src, output_k=10)
different_rotations_of_mask_2 = rotation_augmentation(mask, output_k=10)


# Use-cases (contrast)
different_contrasts_of_src = contrast_augmentation(src)

different_contrasts_of_src_2 = contrast_augmentation(src, output_k=5)

different_contrasts_of_src_3 = contrast_augmentation(src, reduce_contrast=True, output_k=5)
