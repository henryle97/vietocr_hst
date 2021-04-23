from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia

from vietocr.loader.augment_utils import distort, stretch, perspective

def distort_func(images, random_state, parents, hooks):
    for i in range(len(images)):
        images[i] = distort(images[i], 4)

    return images

def stretch_func(images, random_state, parents, hooks):
    for i in range(len(images)):
        images[i] = stretch(images[i], 4)

    return images

def perspective_func(images, random_state, parents, hooks):
    for i in range(len(images)):
        images[i] = perspective(images[i])

    return images

class ImgAugTransform:
  def __init__(self):
    sometimes = lambda aug: iaa.Sometimes(0.3, aug)

    self.aug = iaa.Sequential(iaa.SomeOf((1, 5), 
        [
        # blur

        sometimes(iaa.OneOf([iaa.GaussianBlur(sigma=(0, 1.0)),
                            iaa.MotionBlur(k=3)])),
        
        # color
        sometimes(iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)),
        sometimes(iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True)),
        sometimes(iaa.Invert(0.25, per_channel=0.5)),
        sometimes(iaa.Solarize(0.5, threshold=(32, 128))),
        sometimes(iaa.Dropout2d(p=0.5)),
        sometimes(iaa.Multiply((0.5, 1.5), per_channel=0.5)),
        sometimes(iaa.Add((-40, 40), per_channel=0.5)),

        sometimes(iaa.JpegCompression(compression=(5, 80))),
        
        # distort
        # geographic transformation
        # sometimes(iaa.OneOf([iaa.Lambda(func_images=distort_func),
        #                      iaa.Lambda(func_images=stretch_func),
        #                      iaa.Lambda(func_images=perspective_func),
        #                      ])),
        sometimes(iaa.Crop(percent=(0.01, 0.05), sample_independently=True)),
        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.01))),
        sometimes(iaa.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1), 
#                            rotate=(-5, 5), shear=(-5, 5), 
                            order=[0, 1], cval=(0, 255), 
                            mode=ia.ALL)),
        sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.01))),
        sometimes(iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                            iaa.CoarseDropout(p=(0, 0.1), size_percent=(0.02, 0.25))])),

    ],
        random_order=True),
    random_order=True)
      
  def __call__(self, img):
    img = np.array(img)
    img = self.aug.augment_image(img)
    img = Image.fromarray(img)
    return img


if __name__ == "__main__":
    aug = ImgAugTransform()
    import cv2

    img = cv2.imread("C:/Users/hoanglv10/PycharmProjects/vietocr/sample/001063014772.jpeg")
    img_aug = aug(img)
    import matplotlib.pyplot as plt
    plt.imshow(img_aug)
    plt.show()

