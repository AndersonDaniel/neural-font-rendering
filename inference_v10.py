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
from train_v10_small import load_batch, GLYPH_NUMS, INTENSITY_CATEGORIES


def main():
    model = tf.keras.models.load_model('/data/training/v10_real_small/m/model')
    roots = {
        'arsenica': {
            1/7: '/data/ground_truth/arsenica/arsenicatrial-thin',
            2/7: '/data/ground_truth/arsenica/arsenicatrial-light',
            3/7: '/data/ground_truth/arsenica/arsenicatrial-regular',
            4/7: '/data/ground_truth/arsenica/arsenicatrial-medium',
            5/7: '/data/ground_truth/arsenica/arsenicatrial-demibold',
            6/7: '/data/ground_truth/arsenica/arsenicatrial-bold',
            7/7: '/data/ground_truth/arsenica/arsenicatrial-extrabold'
        },
        'times_new_roman': {
            3/7: '/data/ground_truth/times_new_roman/times_new_roman_regular',
            6/7: '/data/ground_truth/times_new_roman/times_new_roman_bold'
        },
        'roboto': {
            1/7: '/data/ground_truth/roboto/roboto-thin',
            2/7: '/data/ground_truth/roboto/roboto-light',
            3/7: '/data/ground_truth/roboto/roboto-regular',
            4/7: '/data/ground_truth/roboto/roboto-medium',
            6/7: '/data/ground_truth/roboto/roboto-bold',
        },
        'hind': {
            2/7: '/data/ground_truth/hind/hind-light',
            3/7: '/data/ground_truth/hind/hind-regular',
            4/7: '/data/ground_truth/hind/hind-medium',
            5/7: '/data/ground_truth/hind/hind-semibold',
            6/7: '/data/ground_truth/hind/hind-bold',
        },
        'dancing_script': {
            3/7: '/data/ground_truth/dancing_script/dancingscript-regular',
            4/7: '/data/ground_truth/dancing_script/dancingscript-medium',
            5/7: '/data/ground_truth/dancing_script/dancingscript-semibold',
            6/7: '/data/ground_truth/dancing_script/dancingscript-bold',
        },
        'roboto_slab': {
            1/7: '/data/ground_truth/roboto_slab/robotoslab-thin',
            2/7: '/data/ground_truth/roboto_slab/robotoslab-light',
            3/7: '/data/ground_truth/roboto_slab/robotoslab-regular',
            4/7: '/data/ground_truth/roboto_slab/robotoslab-medium',
            5/7: '/data/ground_truth/roboto_slab/robotoslab-semibold',
            6/7: '/data/ground_truth/roboto_slab/robotoslab-bold',
            7/7: '/data/ground_truth/roboto_slab/robotoslab-extrabold'
        }
    }

    results_root = '/data/results/v10_real_small/m/'
    
    comparison_results_path = os.path.join(results_root, 'comparison')

    subdirs = {
        font: {
            k: next(os.walk(v))[1]
            for k, v in font_roots.items()
        }
        for font, font_roots in roots.items()
    }

    for font, font_subdirs in subdirs.items():
        for k, v in font_subdirs.items():
            v.sort(key=lambda x: int(x.split('_')[0]))

    bitmap_size = int(list(list(subdirs.values())[0].values())[0][-1].split('_')[0])
    subdirs = {
        font: {
            k: [os.path.join(roots[font][k], s) for s in v]
            for k, v in font_subdirs.items()
        }
        for font, font_subdirs in subdirs.items()
    }

    glyph_nums = GLYPH_NUMS
    sample_dir = list(list(roots.values())[0].values())[0]
    glyph_metadata = pd.read_csv(os.path.join(*os.path.split(sample_dir)[:-1], 'glyphs.csv'))
    glyph_metadata['modifier_indices'] = glyph_metadata['modifier_indices'].apply(json.loads)
    with open('/data/glyph_names.json', 'r') as f:
        glyph_names = json.load(f)
        base_names = glyph_names['base']
        modifier_names = glyph_names['modifiers']

    errors = []
    errors_idx = []

    for font_idx, (font, font_subdirs) in enumerate(subdirs.items()):
        for e_idx, (e, e_subdirs) in enumerate(font_subdirs.items()):
            for subdir_idx, subdir in tqdm(enumerate(e_subdirs), total=len(e_subdirs)):
                curr_errors = []
                subdir_base = os.path.split(subdir)[-1]
                curr_comparison_results_path = os.path.join(comparison_results_path, str(font_idx), str(e_idx), subdir_base)
                errors_idx.append(f'{font_idx}_{e_idx}_{subdir_base}')
                
                os.makedirs(curr_comparison_results_path, exist_ok=True)


                curr_batch = [
                    (os.path.join(subdir, f'{glyph_num}.png'),
                    glyph_num,
                    subdir_idx, font_idx, e)
                    for glyph_num in glyph_nums
                ]

                for i, sample in enumerate(curr_batch):
                    glyph_num = sample[1]
                    curr_X_glyph, curr_X_size, curr_X_pos, curr_X_font, curr_X_extra, curr_Y, curr_original_Y, _ = load_batch(
                        [sample], len(subdirs), glyph_metadata, bitmap_size, base_names, modifier_names
                    )

                    Y_pred_cat, Y_pred, _ = model([curr_X_glyph, curr_X_size, curr_X_pos, curr_X_font, curr_X_extra])
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
                            index=errors_idx,
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
