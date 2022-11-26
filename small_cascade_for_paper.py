import os
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt


BACKGROUND = (np.array(plt.get_cmap('RdBu')(.5)) * 240).astype(np.uint8)[:3]

COLUMN_PAD = 1

ROOT = '/home/ubuntu/data/results/'
IMPLICIT = os.path.join(ROOT, 'implicit')
MASKED_MLP = os.path.join(ROOT, 'memorization_masked_mlp')


def pad(img, size):
    res = np.ones((size, size + 2 * COLUMN_PAD, 3)) * 255
    # res = np.tile(BACKGROUND, (size, size + 2 * COLUMN_PAD, 1))
    img_size = img.shape[0]
    padding = (size - img_size) // 2
    res[padding:padding + img_size, COLUMN_PAD + padding:COLUMN_PAD + padding + img_size] = img
    res[:, :COLUMN_PAD] = res[:, -COLUMN_PAD:] = BACKGROUND

    return res


def pad_multi(img, size):
    h = img.shape[0]
    n = img.shape[1] // h
    return np.concatenate([pad(img[:, i * h:(i + 1) * h], size) for i in range(n)], axis=1)


def load_implicit(font, exp_name, size, glyph_idx):
    curr_root = os.path.join(IMPLICIT, font, exp_name, 'results', 'comparison')
    dirs = os.listdir(curr_root)
    curr_dir = [x for x in dirs if x.startswith(str(size))][0]
    img = imread(os.path.join(curr_root, curr_dir, f'{glyph_idx}.png'))
    w = img.shape[1] // 3
    
    # return img[:, -w:]
    return img[:, :w]


def load_masked_mlp(font, exp_name, size, glyph_idx):
    curr_root = os.path.join(MASKED_MLP, font, exp_name, 'results', 'comparison')
    dirs = os.listdir(curr_root)
    curr_dir = [x for x in dirs if x.startswith(str(size))][0]
    img = imread(os.path.join(curr_root, curr_dir, f'{glyph_idx}.png'))
    w = img.shape[1] // 3

    return img[:, :w]


def process_font(font):
    images = []
    curr_row = []
    for glyph_idx in range(ord('a'), ord('z') + 1):
        glyph = chr(glyph_idx)
        exp_name = f'lowercase_{glyph}'
        implicit = load_implicit(font, exp_name, 60, glyph_idx)
        masked_mlp = load_masked_mlp(font, exp_name, 60, glyph_idx)

        curr_row.append(np.concatenate([masked_mlp, implicit], axis=0))

    imsave(os.path.join(ROOT, f'{font}_small_cascade.png'),
           np.concatenate(curr_row, axis=1))


def main():
    for font in ['times_new_roman', 'tahoma', 'arial']:
        process_font(font)
        
        # break


if __name__ == '__main__':
    main()
