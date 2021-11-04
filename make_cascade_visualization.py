import os
import numpy as np
from skimage.io import imread, imsave

def main(root):
    indices = sum([
        list(range(ord('A'), ord('Z') + 1)),
        list(range(ord('a'), ord('z') + 1)),
        list(range(ord('0'), ord('9') + 1)),
        list(map(ord, '!@#$%^&*()'))
    ], [])

    dirs = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))]
    dirs_n_sizes = [(int(x.split('_')[0]), x) for x in dirs]
    dirs_n_sizes.sort(key=lambda x: x[0])
    
    bmp_size = dirs_n_sizes[-1][0]
    res = []

    for size, dirname in dirs_n_sizes:
        curr_res = []
        dirpath = os.path.join(root, dirname)
        for cidx in indices:
            img = imread(os.path.join(dirpath, f'{cidx}.png'))
            padding = (bmp_size - size) // 2
            img_full = np.ones((bmp_size, 3 * bmp_size, 3), dtype=img.dtype) * 255
            img_full[padding:padding + size, padding:padding + 3 * size] = img
            curr_res.append(img_full)

        res.append(np.concatenate(curr_res, axis=1))

    res = np.concatenate(res, axis=0)
    imsave('/data/cascade.png', res)


if __name__ == '__main__':
    main('/data/results/v2/times_new_roman/comparison')