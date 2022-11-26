import tensorflow as tf
from train_v1 import load_img, make_larger, make_smaller, make_mask, load_glyph_nums
import numpy as np
import json
import os
import pandas as pd
import imageio


def main():
    model = tf.keras.models.load_model('/data/training/v4_small/model_roboto')
    with open('/data/glyph_names.json', 'r') as f:
            glyph_names = json.load(f)
            base_names = glyph_names['base']
            modifier_names = glyph_names['modifiers']

    n_sizes, glyph_metadata, bitmap_size = get_misc_stuff()
    chr2render = 'M'
    size_idx = n_sizes // 2
    target_bmp_size = bitmap_size - (n_sizes - size_idx - 1)
    n_weights = 60

    glyph_record = glyph_metadata[glyph_metadata['idx'] == ord(chr2render)]
    x_glyph_base = np.zeros(len(base_names))
    x_glyph_base[glyph_record['base_name_idx'].values[0]] = 1
    x_glyph_modifiers = np.zeros(len(modifier_names))
    glyph_modifiers = glyph_record['modifier_indices'].values[0]
    if len(glyph_modifiers) > 0:
            x_glyph_modifiers[glyph_modifiers] = 1

    x_size = np.zeros(n_sizes)
    x_size[size_idx] = 1
    x = np.concatenate([x_glyph_base, x_glyph_modifiers, x_size, [0]])
    X = np.tile(x, (n_weights, 1))
    X[:, -1] = np.linspace(.1, .9, n_weights)

    _, downscale_idx = make_larger(np.zeros((target_bmp_size, target_bmp_size)), bitmap_size)
    m = make_mask(downscale_idx, bitmap_size)
    Y = model.predict(X)[:, np.where(m.ravel())].reshape((-1, target_bmp_size, target_bmp_size))
    Y = ((1 - Y) * 255).astype(np.uint8)
    images = [Y[i] for i in range(Y.shape[0])]
    with imageio.get_writer('/data/weight_animation.mp4', fps=30) as writer:
        for img in images:
            writer.append_data(img)
    

def get_misc_stuff():
    root = '/data/ground_truth/roboto/roboto-thin'
    subdirs = next(os.walk(root))[1]
    subdirs.sort(key=lambda x: int(x.split('_')[0]))
    bitmap_size = int(subdirs[-1].split('_')[0])
    subdirs = [os.path.join(root, s) for s in subdirs]
    glyph_nums = load_glyph_nums(root)
    glyph_metadata = pd.read_csv(os.path.join(*os.path.split(root)[:-1], 'glyphs.csv'))
    glyph_metadata['modifier_indices'] = glyph_metadata['modifier_indices'].apply(json.loads)

    return len(subdirs), glyph_metadata, bitmap_size




if __name__ == '__main__':
    main()

