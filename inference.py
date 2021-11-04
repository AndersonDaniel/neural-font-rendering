import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from glob import glob
import tensorflow as tf
import cv2
import pandas as pd
import json
import matplotlib.pyplot as plt
from train_v1 import load_img, make_larger, make_smaller, make_mask, load_batch, load_glyph_nums


def main():
    model = tf.keras.models.load_model('/data/training/v1/model')
    root = '/data/ground_truth/times_new_roman'
    results_root = '/data/results/v2/times_new_roman'
    raw_results_path = os.path.join(results_root, 'raw')
    comparison_results_path = os.path.join(results_root, 'comparison')

    subdirs = next(os.walk(root))[1]
    subdirs.sort(key=lambda x: int(x.split('_')[0]))
    bitmap_size = int(subdirs[-1].split('_')[0])
    subdirs = [os.path.join(root, s) for s in subdirs]
    glyph_nums = load_glyph_nums(root)
    glyph_metadata = pd.read_csv(os.path.join(root, 'glyphs.csv'))
    glyph_metadata['modifier_indices'] = glyph_metadata['modifier_indices'].apply(json.loads)
    with open('/data/glyph_names.json', 'r') as f:
        glyph_names = json.load(f)
        base_names = glyph_names['base']
        modifier_names = glyph_names['modifiers']

    errors = []

    for subdir_idx, subdir in tqdm(enumerate(subdirs), total=len(subdirs)):
        curr_errors = []
        subdir_base = os.path.split(subdir)[-1]
        curr_raw_results_path = os.path.join(raw_results_path, subdir_base)
        curr_comparison_results_path = os.path.join(comparison_results_path, subdir_base)
        os.makedirs(curr_raw_results_path, exist_ok=True)
        os.makedirs(curr_comparison_results_path, exist_ok=True)

        curr_batch = [
            (os.path.join(subdir, f'{glyph_num}.png'),
             glyph_num,
             subdir_idx)
            for glyph_num in glyph_nums
        ]

        curr_X, curr_M, curr_Y = load_batch(
            curr_batch, len(subdirs), glyph_metadata, bitmap_size, base_names, modifier_names
        )

        Y_pred = model.predict(curr_X)

        for i, (_, glyph_num, _) in enumerate(curr_batch):
            img = Y_pred[i][np.where(curr_M[i])].reshape((-1, int(np.sqrt(curr_M[i].sum()))))
            img = Image.fromarray(((1 - img) * 255).astype(np.uint8), 'L')
            img_raw = Y_pred[i].reshape((bitmap_size, -1))
            img_raw = Image.fromarray(((1 - img_raw) * 255).astype(np.uint8), 'L')

            err_img, gt_img, curr_error = get_error_image(Y_pred[i], curr_M[i], curr_Y[i])
            
            res_img = h_concat_images(img, err_img, gt_img)

            img_raw.save(os.path.join(curr_raw_results_path, f'{glyph_num}.png'))
            res_img.save(os.path.join(curr_comparison_results_path, f'{glyph_num}.png'))
            # curr_error = ((np.asarray(img) - np.asarray(gt_img)) ** 2).mean()
            curr_errors.append(curr_error)

        errors.append(curr_errors)

    error_df = pd.DataFrame(errors,
                            index=[os.path.split(s)[-1] for s in subdirs],
                            columns=[chr(g) for g in glyph_nums]
    )
    
    error_df.to_csv(os.path.join(results_root, 'errors.csv'))


def get_error_image(y_pred, m, y):
    y_pred_expanded = y_pred[np.where(m)].reshape((-1, int(np.sqrt(m.sum()))))
    y_expanded = y[np.where(m)].reshape((-1, int(np.sqrt(m.sum()))))
    diff = y_pred_expanded - y_expanded
    cmap = plt.get_cmap('RdBu')
    img = cmap((diff + 1) / 2)
    return (Image.fromarray(np.uint8(img * 255), 'RGBA'),
            Image.fromarray(((1 - y_expanded) * 255).astype(np.uint8), 'L'),
            (diff ** 2).mean())


def h_concat_images(*imgs):
    res = Image.new('RGB', (sum(img.width for img in imgs), imgs[0].height))
    curr_pos = 0
    for img in imgs:
        res.paste(img, (curr_pos, 0))
        curr_pos += img.width

    return res


if __name__ == '__main__':
    main()
