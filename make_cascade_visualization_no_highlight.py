import os
import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm
import pandas as pd


def main(root):
    indices = sum([
        list(range(ord('A'), ord('Z') + 1)),
        list(range(ord('a'), ord('z') + 1)),
        list(range(ord('0'), ord('9') + 1)),
        list(map(ord, '!@#$%^&*()'))
    ], [])

    above_root = os.path.join(*os.path.split(root)[:-1])
    indices_chars = list(map(chr, indices))
    indices = list(map(ord, indices_chars))

    levels = sorted([x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))])
    sublevels = [
        sorted([x for x in os.listdir(os.path.join(root, level)) if os.path.isdir(os.path.join(root, level, x))])
        for level in levels
    ]
    
    dirs = [x for x in os.listdir(os.path.join(root, levels[0], sublevels[0][0]))
            if os.path.isdir(os.path.join(root, levels[0], sublevels[0][0], x))]

    dirs_n_sizes = [(int(x.split('_')[0]), x) for x in dirs]
    dirs_n_sizes.sort(key=lambda x: x[0])
    
    bmp_size = dirs_n_sizes[-1][0]
    res = []

    for size, _ in tqdm(dirs_n_sizes, desc=root):
        curr_res = []
        for level_idx, level in enumerate(levels):
            for sublevel in sublevels[level_idx]:
                dirname = [x for x in os.listdir(os.path.join(root, level, sublevel))
                           if x.startswith(f'{size}_')][0]
                dirpath = os.path.join(root, level, sublevel, dirname)
                for cidx in indices:
                    err_idx = f'{level}_{sublevel}_{dirname}'
                    try:
                        img = imread(os.path.join(dirpath, f'{cidx}.png'))
                    except:
                        continue

                    padding = (bmp_size - size) // 2
                    img_full = np.ones((3 * bmp_size, bmp_size, 3), dtype=img.dtype) * 255
                    img_full[padding:padding + size, padding:padding + size] = img[:, :size]
                    img_full[padding + size:padding + 2 * size, padding:padding + size] = img[:, size:2 * size]
                    img_full[padding + 2 * size:padding + 3 * size, padding:padding + size] = \
                        img[:, 2 * size:3 * size]

                    curr_res.append(img_full)

        res.append(np.concatenate(curr_res, axis=1))

    res = np.concatenate(res, axis=0)
    imsave(os.path.join(*os.path.split(root)[:-1], 'cascade.png'), res)


if __name__ == '__main__':
    main('/data/results/v11_real_small/cap_a_interpolated_fonts/comparison')
    # main('/data/results/v10_real_small/cap_a/comparison')
    # main('/data/results/v9_real_small/cap_a_ampercent/arsenica/comparison', two_levels=True)
    # main('/data/results/v8_real_small/abcxyz/times_new_roman/comparison')
    # main('/data/results/v6_small/times_new_roman/comparison')
    # main('/data/results/v2_smaller/times_new_roman/comparison')
    # main('/data/results/v4_smaller/roboto/roboto-thin/comparison')
    # main('/data/results/v4_smaller/roboto/roboto-light/comparison')
    # main('/data/results/v4_smaller/roboto/roboto-regular/comparison')
    # main('/data/results/v4_smaller/roboto/roboto-medium/comparison')
    # main('/data/results/v4_smaller/roboto/roboto-bold/comparison')