import matplotlib.pyplot as plt
import numpy as np
import cv2


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    # if ax is None:
    #     fig, ax = plt.subplots()
    # image = image.transpose((1, 2, 0))

    # if normalize:
    #     mean = np.array([0.485, 0.456, 0.406])
    #     std = np.array([0.229, 0.224, 0.225])
    #     image = std * image + mean
    #     image = np.clip(image, 0, 1)

    image = cv2.resize(image, (84, 84))

    # image = image.transpose(2, 1, 0)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # ax.imshow(image)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.tick_params(axis='both', length=0)
    # ax.set_xticklabels('')
    # ax.set_yticklabels('')

    return image
