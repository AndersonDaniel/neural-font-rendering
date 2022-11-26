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
from train_v11_small import GLYPH_NUMS, INTENSITY_CATEGORIES, positional_encodings


def load_random_batch(n_sizes, bitmap_size, size_idx):
    X_size = []
    X_pos = []
    
    relative_size = size_idx / n_sizes
    x_size = np.concatenate([[relative_size],
                                positional_encodings(relative_size)
                                ])
    for row in range(bitmap_size):
        vert_pos = -1 + (2 * row + 1) / bitmap_size
        vert_encodings = positional_encodings(vert_pos)
        for col in range(bitmap_size):
            hori_pos = -1 + (2 * col + 1) / bitmap_size
            hori_encodings = positional_encodings(hori_pos)
            x_pos = np.concatenate([[vert_pos, hori_pos],
                                    vert_encodings, hori_encodings])
            
            X_size.append(x_size)
            X_pos.append(x_pos)
    

    X_size = np.stack(X_size)
    X_pos = np.stack(X_pos)

    return X_size, X_pos


def main():
    model = tf.keras.models.load_model('/data/training/v11_real_small/cap_a/inner_model')
    embeddings = tf.keras.models.load_model('/data/training/v11_real_small/cap_a/font_embeddings')

    results_root = '/data/results/v11_real_small/cap_a_interpolated_fonts/'
    
    comparison_results_path = os.path.join(results_root, 'comparison')

    MIN_BMP_SIZE = 20
    MAX_BMP_SIZE = 63

    glyph_nums = GLYPH_NUMS
    EXTRA = 3 / 7
    INTERPOLATION_WEIGHTS = np.linspace(0, 1, 7).tolist()
    combinations = [
        (0, 5), (0, 2), (0, 4),
        (3, 4), (2, 5)
    ]

    for pair_idx, (font1_idx, font2_idx) in enumerate(combinations):
        font1_embedding = embeddings(font1_idx)
        font2_embedding = embeddings(font2_idx)
        for interpolation_weight_idx, interpolation_weight in enumerate(INTERPOLATION_WEIGHTS):
            font_embedding = tf.reshape(font1_embedding * (1 - interpolation_weight)
                                        + font2_embedding * interpolation_weight, (1, -1))
            for size_idx in tqdm(range(MAX_BMP_SIZE - MIN_BMP_SIZE + 1)):
                curr_size = size_idx + MIN_BMP_SIZE
                curr_comparison_results_path = os.path.join(comparison_results_path,
                                                            str(pair_idx),
                                                            str(interpolation_weight_idx), f'{curr_size}_foo')

                X_size, X_pos = load_random_batch(MAX_BMP_SIZE - MIN_BMP_SIZE + 1, curr_size, size_idx)
                curr_original_Y = np.zeros((curr_size * curr_size))
                
                os.makedirs(curr_comparison_results_path, exist_ok=True)

                X_font = tf.repeat(font_embedding, repeats=X_size.shape[0], axis=0)
                X_extra = EXTRA * tf.ones((X_size.shape[0], 1))

                Y_pred_cat, Y_pred, _ = model([X_size, X_pos, X_font, X_extra])
                Y_pred = Y_pred.numpy()
                
                img = Y_pred.reshape((-1, int(np.sqrt(Y_pred.shape[0]))))

                img = Image.fromarray(((1 - img) * 255).astype(np.uint8), 'L')

                err_img, gt_img, curr_error = get_error_image(Y_pred, curr_original_Y)
                
                res_img = h_concat_images(img, err_img, gt_img)

                res_img.save(os.path.join(curr_comparison_results_path, f'{GLYPH_NUMS[0]}.png'))


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
