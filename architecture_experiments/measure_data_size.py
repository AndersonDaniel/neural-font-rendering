from skimage.io import imread
import os
import numpy as np


def main():
    root = '/home/ubuntu/data/ground_truth/times_new_roman/'
    subdirs = [
        os.path.join(root, x) for x in os.listdir(root)
    ]

    subdirs = [x for x in subdirs if os.path.isdir(x)]

    total_size = 0
    for subdir in subdirs:
        img = imread(os.path.join(subdir, '103.png'))

        total_size += np.product(img.shape)

    print(total_size)


if __name__ == '__main__':
    main()
