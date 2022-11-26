import os
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt


BACKGROUND = (np.array(plt.get_cmap('RdBu')(.5)) * 240).astype(np.uint8)[:3]

COLUMN_PAD = 1

MIN_SIZE = 20
MAX_SIZE = 63

ROOT = '/home/ubuntu/data/results/'
IMPLICIT_MULTIPLE_WEIGHTS = os.path.join(ROOT, 'implicit_multiple_weights')


def pad(img, size):
    res = np.ones((size, size)) * 255
    # res = np.tile(BACKGROUND, (size, size + 2 * COLUMN_PAD, 1))
    img_size = img.shape[0]
    padding = (size - img_size) // 2
    res[padding:padding + img_size, padding:padding + img_size] = img

    return res


def pad_multi(img, size):
    h = img.shape[0]
    n = img.shape[1] // h
    return np.concatenate([pad(img[:, i * h:(i + 1) * h], size) for i in range(n)], axis=1)


def load_results(font, exp_name, size, glyph_idx):
    curr_root = os.path.join(IMPLICIT_MULTIPLE_WEIGHTS, font, exp_name, 'weight_interpolation')
    
    img = imread(os.path.join(curr_root, f'weight_interpolation_{size}.png'))
    w = img.shape[1] // 3
    
    return pad_multi(img, MAX_SIZE)


def process_font(font):
    images = []
    for size in range(MIN_SIZE, MAX_SIZE + 1):
        curr_row = []
        for glyph_idx in range(ord('a'), ord('e') + 1):
            glyph = chr(glyph_idx)
            exp_name = f'lower_case_{glyph}'
            res = load_results(font, exp_name, size, glyph_idx).astype(np.uint8)

            curr_row.append(res)
            curr_row.append(np.ones((MAX_SIZE, 2)).astype(np.uint8) * 200)

        images.append(np.concatenate(curr_row, axis=1))

    imsave(os.path.join(ROOT, f'{font}_weight_interpolation_cascade.png'),
           np.concatenate(images, axis=0))


def main():
    process_font('roboto')


if __name__ == '__main__':
    main()
