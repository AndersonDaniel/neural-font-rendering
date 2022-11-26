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
from train_v10_small import GLYPH_NUMS, INTENSITY_CATEGORIES, positional_encodings


def main():
    model = tf.keras.models.load_model('/data/training/v10_real_small/cap_a/model')
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
            1/7: None,
            2/7: None,
            3/7: '/data/ground_truth/times_new_roman/times_new_roman_regular',
            4/7: None,
            5/7: None,
            6/7: '/data/ground_truth/times_new_roman/times_new_roman_bold',
            7/7: None
        },
        'roboto': {
            1/7: '/data/ground_truth/roboto/roboto-thin',
            2/7: '/data/ground_truth/roboto/roboto-light',
            3/7: '/data/ground_truth/roboto/roboto-regular',
            4/7: '/data/ground_truth/roboto/roboto-medium',
            5/7: None,
            6/7: '/data/ground_truth/roboto/roboto-bold',
            7/7: None
        },
        'hind': {
            1/7: None,
            2/7: '/data/ground_truth/hind/hind-light',
            3/7: '/data/ground_truth/hind/hind-regular',
            4/7: '/data/ground_truth/hind/hind-medium',
            5/7: '/data/ground_truth/hind/hind-semibold',
            6/7: '/data/ground_truth/hind/hind-bold',
            7/7: None
        },
        'dancing_script': {
            1/7: None,
            2/7: None,
            3/7: '/data/ground_truth/dancing_script/dancingscript-regular',
            4/7: '/data/ground_truth/dancing_script/dancingscript-medium',
            5/7: '/data/ground_truth/dancing_script/dancingscript-semibold',
            6/7: '/data/ground_truth/dancing_script/dancingscript-bold',
            7/7: None
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

    results_root = '/data/results/v10_real_small/cap_a_full_weights/'
    
    comparison_results_path = os.path.join(results_root, 'comparison')

    subdirs = {
        font: {
            k: (next(os.walk(v))[1] if v is not None else None)
            for k, v in font_roots.items()
        }
        for font, font_roots in roots.items()
    }

    for font, font_subdirs in subdirs.items():
        for k, v in font_subdirs.items():
            if v is None:
                continue

            v.sort(key=lambda x: int(x.split('_')[0]))

    bitmap_size = int(list(list(subdirs.values())[0].values())[0][-1].split('_')[0])
    subdirs = {
        font: {
            k: ([os.path.join(roots[font][k], s) for s in v] if v is not None else None)
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
    temp_subdirs = None

    for font_idx, (font, font_subdirs) in enumerate(subdirs.items()):
        for e_idx, (e, e_subdirs) in enumerate(font_subdirs.items()):
            if e_subdirs is None:
                e_subdirs = [f'{x}foo' for x in temp_subdirs]
            else:
                temp_subdirs = e_subdirs

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


def load_batch(batch, n_sizes, glyph_metadata, bitmap_size, base_names, modifier_names):
    X_glyph = []
    X_size = []
    X_pos = []
    X_font = []
    X_extra = []
    Y = []
    W = []
    for img_path, ordered_glyph_idx, size_idx, font_idx, extra in batch:
        try:
            y = load_img(img_path)
        except:
            curr_size = int(img_path.split('/')[-2].split('_')[0])
            y = np.zeros((curr_size, curr_size))

        curr_w = 1.
        if np.random.random() <= -1:
            extra += np.random.normal(scale=.01)
            curr_w = .2

        glyph_record = glyph_metadata[glyph_metadata['idx'] == ordered_glyph_idx]
        x_glyph_base = [glyph_record.index.values[0]]

        relative_size = size_idx / n_sizes
        x_size = np.concatenate([[relative_size],
                                 positional_encodings(relative_size)
                                 ])
        height = y.shape[0]
        width = y.shape[1]
        for row in range(height):
            vert_pos = -1 + (2 * row + 1) / height
            vert_encodings = positional_encodings(vert_pos)
            for col in range(width):
                hori_pos = -1 + (2 * col + 1) / width
                hori_encodings = positional_encodings(hori_pos)
                x_pos = np.concatenate([[vert_pos, hori_pos],
                                        vert_encodings, hori_encodings])
                
                X_glyph.append(x_glyph_base)
                X_size.append(x_size)
                X_pos.append(x_pos)
                X_font.append(np.array([font_idx]))
                X_extra.append(np.array([extra]))
                W.append(curr_w)
        
        Y += y.ravel().tolist()

    X_glyph = np.stack(X_glyph)
    X_size = np.stack(X_size)
    X_pos = np.stack(X_pos)
    X_font = np.stack(X_font)
    X_extra = np.stack(X_extra)
    Y = np.stack(Y)
    W = np.stack(W).astype(np.float32)

    categorical_Y = np.zeros((Y.shape[0], INTENSITY_CATEGORIES))
    categorical_Y[range(Y.shape[0]), (np.floor(Y / (1 / (INTENSITY_CATEGORIES - 1)))).astype(int)] = 1

    return X_glyph, X_size, X_pos, X_font, X_extra, categorical_Y, Y, W


if __name__ == '__main__':
    main()
