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
from train_v9_small import load_batch, load_glyph_nums, INTENSITY_CATEGORIES


def main():
    model = tf.keras.models.load_model('/data/training/v9_real_small/cap_a_ampercent/model_arsenica')
    roots = {
        1/7: '/data/ground_truth/arsenica/arsenicatrial-thin',
        2/7: '/data/ground_truth/arsenica/arsenicatrial-light',
        3/7: '/data/ground_truth/arsenica/arsenicatrial-regular',
        4/7: '/data/ground_truth/arsenica/arsenicatrial-medium',
        5/7: '/data/ground_truth/arsenica/arsenicatrial-demibold',
        6/7: '/data/ground_truth/arsenica/arsenicatrial-bold',
        7/7: '/data/ground_truth/arsenica/arsenicatrial-extrabold'
    }

    results_root = '/data/results/v9_real_small/cap_a_ampercent/arsenica'
    
    comparison_results_path = os.path.join(results_root, 'comparison')

    subdirs = {
        k: next(os.walk(v))[1]
        for k, v in roots.items()
    }

    for k, v in subdirs.items():
        v.sort(key=lambda x: int(x.split('_')[0]))
        # subdirs[k] = [v[10]]

    bitmap_size = int(list(subdirs.values())[0][-1].split('_')[0])
    subdirs = {
        k: [os.path.join(roots[k], s) for s in v]
        for k, v in subdirs.items()
    }

    glyph_nums = load_glyph_nums(list(roots.values())[0])
    glyph_metadata = pd.read_csv(os.path.join(*os.path.split(list(roots.values())[0])[:-1], 'glyphs.csv'))
    glyph_metadata['modifier_indices'] = glyph_metadata['modifier_indices'].apply(json.loads)
    with open('/data/glyph_names.json', 'r') as f:
        glyph_names = json.load(f)
        base_names = glyph_names['base']
        modifier_names = glyph_names['modifiers']

    errors = []

    for e_idx, (e, e_subdirs) in enumerate(subdirs.items()):
        for subdir_idx, subdir in tqdm(enumerate(e_subdirs), total=len(e_subdirs)):
            curr_errors = []
            subdir_base = os.path.split(subdir)[-1]
            curr_comparison_results_path = os.path.join(comparison_results_path, str(e_idx), subdir_base)
            os.makedirs(curr_comparison_results_path, exist_ok=True)

            curr_batch = [
                (os.path.join(subdir, f'{glyph_num}.png'),
                glyph_num,
                subdir_idx, e)
                for glyph_num in glyph_nums
            ]

            for i, sample in enumerate(curr_batch):
                glyph_num = sample[1]
                curr_X_glyph, curr_X_size, curr_X_pos, curr_X_extra, curr_Y, curr_original_Y, _ = load_batch(
                    [sample], len(subdirs), glyph_metadata, bitmap_size, base_names, modifier_names
                )

                Y_pred_cat, Y_pred, _ = model([curr_X_glyph, curr_X_size, curr_X_pos, curr_X_extra])
                Y_pred = Y_pred.numpy()
                # Y_pred = Y_pred_cat.numpy().argmax(axis=1).astype(float) / (INTENSITY_CATEGORIES - 1)
                img = Y_pred.reshape((-1, int(np.sqrt(Y_pred.shape[0]))))

                img = Image.fromarray(((1 - img) * 255).astype(np.uint8), 'L')

                err_img, gt_img, curr_error = get_error_image(Y_pred, curr_original_Y)
                
                res_img = h_concat_images(img, err_img, gt_img)

                res_img.save(os.path.join(curr_comparison_results_path, f'{glyph_num}.png'))
                curr_errors.append(curr_error)

            errors.append(curr_errors)

    error_df = pd.DataFrame(errors,
                            index=[f'{e_idx}_{os.path.split(s)[-1]}' for e_idx, e_subdirs in enumerate(subdirs.values())
                            for s in e_subdirs],
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
