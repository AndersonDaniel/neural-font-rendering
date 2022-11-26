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
from train import load_img, INTENSITY_CATEGORIES, positional_encodings
from skimage.io import imsave
import imageio


def main(EXP_NAME, BITMAP_SIZE):
    model_path = os.path.join('/home/ubuntu/data/results/implicit_multiple_weights',
                                      'roboto', EXP_NAME, 'model')
    # model_path = os.path.join('/home/ubuntu/data/results/implicit_multiple_weights',
                                    #   'arsenica', EXP_NAME, 'model')
    model = tf.keras.models.load_model(model_path)
    weight_embedding_model_path = os.path.join('/home/ubuntu/data/results/implicit_multiple_weights',
                                      'roboto', EXP_NAME, 'weight_embedding_model')
    # weight_embedding_model_path = os.path.join('/home/ubuntu/data/results/implicit_multiple_weights',
                                    #   'arsenica', EXP_NAME, 'weight_embedding_model')
    weight_embedding_model = tf.keras.models.load_model(weight_embedding_model_path)

    subdirs = os.listdir(os.path.join(
        '/home/ubuntu/data/results/implicit_multiple_weights', 'roboto',
        EXP_NAME, 'results', 'comparison', '0', '0'
    ))

    all_sizes = sorted([int(subdir.split('_')[0]) for subdir in subdirs])

    x_weight = tf.convert_to_tensor(np.array([
        [4/7], [6/7]
        # [2/7], [6/7]
        # [3/7], [6/7]
        # [2/7], [3/7]
        # [1/7], [3/7]
    ]))

    w1, w2 = weight_embedding_model(x_weight, training=True)

    curr_X_glyph, curr_X_size, curr_X_pos, curr_X_font = make_batch(
        all_sizes, BITMAP_SIZE
    )

    images = []

    frames_path = f'/home/ubuntu/data/results/implicit_multiple_weights/roboto/{EXP_NAME}/weight_interpolation'
    # frames_path = f'/home/ubuntu/data/results/implicit_multiple_weights/arsenica/{EXP_NAME}/weight_interpolation'
    os.makedirs(frames_path,
                exist_ok=True)

    intensities = []

    for idx, t in tqdm(list(enumerate(np.linspace(0, 1, 120)))):
        curr_weight_embedding = tf.broadcast_to(tf.reshape((1 - t) * w1 + t * w2, (1, -1)),
                                                (curr_X_size.shape[0], w1.shape[0]))

        Y_pred_cat, Y_pred, _ = model([curr_X_glyph, curr_X_size, curr_X_pos, curr_X_font, curr_weight_embedding],
                                      training=True)

        Y_pred = ((1 - Y_pred.numpy().reshape((BITMAP_SIZE, BITMAP_SIZE))) * 255).astype(np.uint8)

        images.append(Y_pred)
        intensities.append(Y_pred.mean())

        # imsave(f'/home/ubuntu/data/results/implicit_multiple_weights/roboto/{EXP_NAME}/weight_interpolation/{idx:06d}.png', Y_pred)

    intensities = np.array(intensities)
    idx = np.argmin(np.abs(intensities - (intensities[0] + intensities[-1]) / 2))
    
    comparison_image = np.concatenate([images[0], images[idx], images[-1]], axis=1)

    imsave(f'/home/ubuntu/data/results/implicit_multiple_weights/roboto/{EXP_NAME}/weight_interpolation/weight_interpolation_{BITMAP_SIZE}.png', comparison_image)
    # imsave(f'/home/ubuntu/data/results/implicit_multiple_weights/arsenica/{EXP_NAME}/weight_interpolation/weight_interpolation.png', comparison_image)


    # mean_weights = [image.mean() for image in images]

    # with open(f'/home/ubuntu/data/results/implicit_multiple_weights/roboto/{EXP_NAME}/weight_inter.json', 'w') as f:
    #     json.dump(mean_weights, f)

    # video_path = f'/home/ubuntu/data/results/implicit_multiple_weights/roboto/{EXP_NAME}/weight_animation.mp4'
    # with imageio.get_writer(video_path, fps=60) as writer:
    #     for img in images:
    #         writer.append_data(cv2.resize(img, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST))


    # gif_path = f'/home/ubuntu/data/results/implicit_multiple_weights/roboto/{EXP_NAME}/weight_animation.gif'
    # with imageio.get_writer(gif_path, fps=60) as writer:
    #     for img in images:
    #         writer.append_data(img)

    # os.system(f'ffmpeg -framerate 60 -i {frames_path}/%06d.png -c:v copy {video_path}')


def make_batch(all_sizes, bitmap_size):
    n_sizes = len(all_sizes)
    X_glyph = []
    X_size = []
    X_pos = []
    X_font = []
    size_idx = all_sizes.index(bitmap_size)
    x_glyph_base = [0]
    font_idx = 0
    relative_size = size_idx / n_sizes
    x_size = np.concatenate([[relative_size],
                                positional_encodings(relative_size)
                                ])
    height = width = bitmap_size
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

    X_glyph = np.stack(X_glyph)
    X_size = np.stack(X_size)
    X_pos = np.stack(X_pos)
    X_font = np.stack(X_font)

    return X_glyph, X_size, X_pos, X_font


if __name__ == '__main__':
    # for glyph in ['a', 'g', 'm', 't']:
    for glyph in ['a', 'b', 'c', 'd', 'e']:
        for bmp_size in range(20, 64):
            main(f'lower_case_{glyph}', bmp_size)
