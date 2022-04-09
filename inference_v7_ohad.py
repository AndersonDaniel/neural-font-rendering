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
from train_v1 import load_img
from train_v7_small import load_batch, load_glyph_nums, INTENSITY_CATEGORIES


def main():
    model = tf.keras.models.load_model('/data/training/v7_real_small/abcxyz/model_times_new_roman')
    root = '/data/ground_truth/times_new_roman'
    results_root = '/data/results/v7_real_small/abcxyz_generalization_ohad/times_new_roman'
    
    comparison_results_path = os.path.join(results_root, 'comparison')

    subdirs = next(os.walk(root))[1]
    subdirs.sort(key=lambda x: int(x.split('_')[0]))
    # subdirs = subdirs[::5]
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
        curr_comparison_results_path = os.path.join(comparison_results_path, subdir_base)
        os.makedirs(curr_comparison_results_path, exist_ok=True)

        curr_batch = [
            (os.path.join(subdir, f'{glyph_num}.png'),
             glyph_num,
             subdir_idx // 5)
            for glyph_num in glyph_nums
        ]

        for i, sample in enumerate(curr_batch):
            glyph_num = sample[1]
            curr_X_glyph, curr_X_size, curr_X_pos, curr_Y, curr_original_Y = load_batch(
                [sample], int(np.ceil(len(subdirs) / 5)), glyph_metadata, bitmap_size, base_names, modifier_names
            )

            Y_pred = model([curr_X_glyph, curr_X_size, curr_X_pos]).numpy().argmax(axis=1).astype(float) / INTENSITY_CATEGORIES
            img = Y_pred.reshape((-1, int(np.sqrt(Y_pred.shape[0]))))

            img = Image.fromarray(((1 - img) * 255).astype(np.uint8), 'L')

            err_img, gt_img, curr_error = get_error_image(Y_pred, curr_original_Y)
            
            res_img = h_concat_images(img, err_img, gt_img)

            res_img.save(os.path.join(curr_comparison_results_path, f'{glyph_num}.png'))
            curr_errors.append(curr_error)

        errors.append(curr_errors)

    error_df = pd.DataFrame(errors,
                            index=[os.path.split(s)[-1] for s in subdirs],
                            columns=[chr(g) for g in glyph_nums]
    )
    
    error_df.to_csv(os.path.join(results_root, 'errors.csv'))


def get_error_image(y_pred, y):
    y_pred_expanded = y_pred.reshape((-1, int(np.sqrt(y_pred.shape[0]))))
    y_expanded = y.reshape((-1, int(np.sqrt(y.shape[0]))))
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
