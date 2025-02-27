import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import logging
import PIL.Image
import numpy as np


def get_mask_transform(img_size=(224,224)):
    transform_list = []
    transform_list.append(transforms.Resize(img_size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def get_transform(img_size=(224,224), for_val=False):
    transform_list = []

    if for_val:
        transform_list.append(transforms.Resize(img_size))
        transform_list.append(transforms.CenterCrop(img_size))
        transform_list.append(transforms.ToTensor())
    else:
        transform_list.append(transforms.Resize(img_size))
        transform_list.append(transforms.CenterCrop(img_size))
        transform_list.append(AllAugmentations())
        transform_list.append(transforms.ToTensor())

    transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    transform = transforms.Compose(transform_list)
    return transform




class AllAugmentations(object):
    def __init__(self):
        import albumentations
        self.transform = albumentations.Compose([
            albumentations.Blur(blur_limit=3),
            albumentations.JpegCompression(quality_lower=30, quality_upper=100, p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.HueSaturationValue(p=0.5),
            albumentations.GridDistortion(num_steps=5,distort_limit=0.1, p=0.2),
            albumentations.ElasticTransform(alpha=0.5,sigma=10,alpha_affine=10, p=0.2),
            albumentations.Perspective(scale=(0.00, 0.02), p=0.2),
            albumentations.OpticalDistortion(distort_limit=0.01, shift_limit=0.01, p=0.2),
            albumentations.RandomSunFlare(flare_roi=(0, 0, 1, 0.2), angle_lower=0.5, num_flare_circles_lower=2,num_flare_circles_upper=4,src_radius=40, p=0.2),  
            albumentations.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, alpha_coef=0.75, p=0.2),
            albumentations.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=2, p=0.2),
            albumentations.MotionBlur(blur_limit=3,p=0.5),
            albumentations.ColorJitter(brightness=0.4, p=0.5),
            albumentations.RandomGamma(gamma_limit=(80, 120)),
            albumentations.CLAHE(),
        ])
    def __call__(self, image):
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        image_pil = PIL.Image.fromarray(augmented['image'])
        return image_pil

class JPEGCompression(object):
    def __init__(self, level):
        import albumentations as A
        self.level = level
        self.transform = A.augmentations.transforms.JpegCompression(p=1)

    def __call__(self, image):
        image_np = np.array(image)
        image_out = self.transform.apply(image_np, quality=self.level)
        image_pil = PIL.Image.fromarray(image_out)
        return image_pil

class Blur(object):
    def __init__(self, level):
        import albumentations as A
        self.level = level
        self.transform = A.Blur(blur_limit=(self.level, self.level), always_apply=True)

    def __call__(self, image):
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        image_pil = PIL.Image.fromarray(augmented['image'])
        return image_pil

class Gamma(object):
    def __init__(self, level):
        import albumentations as A
        self.level = level
        self.transform = A.augmentations.transforms.RandomGamma(p=1)

    def __call__(self, image):
        image_np = np.array(image)
        image_out = self.transform.apply(image_np, gamma=self.level/100)
        image_pil = PIL.Image.fromarray(image_out)
        return image_pil

