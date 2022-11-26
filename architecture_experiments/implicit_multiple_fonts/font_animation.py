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
from train import load_img, GLYPH_NUMS, INTENSITY_CATEGORIES, positional_encodings, EXP_NAME
from skimage.io import imsave
import imageio


BITMAP_SIZE = 62


def main():
    model_path = os.path.join('/home/ubuntu/data/results/implicit_multiple_fonts',
                                      'exp2', EXP_NAME, 'model')
    model = tf.keras.models.load_model(model_path)
    extra_embedding_model_path = os.path.join('/home/ubuntu/data/results/implicit_multiple_fonts',
                                      'exp2', EXP_NAME, 'extra_embedding_model')
    extra_embedding_model = tf.keras.models.load_model(extra_embedding_model_path)

    subdirs = os.listdir(os.path.join(
        '/home/ubuntu/data/results/implicit_multiple_fonts', 'exp2',
        EXP_NAME, 'results', 'comparison', '0', '0'
    ))

    all_sizes = sorted([int(subdir.split('_')[0]) for subdir in subdirs])

    x_weight = tf.convert_to_tensor(np.array([
        [-1], [1]
        # [2/7], [6/7]
    ]))

    e1, e2 = extra_embedding_model(x_weight, training=True)

    curr_X_glyph, curr_X_size, curr_X_pos, curr_X_font = make_batch(
        all_sizes, BITMAP_SIZE
    )

    images = []

    frames_path = f'/home/ubuntu/data/results/implicit_multiple_fonts/exp2/{EXP_NAME}/font_interpolation'
    os.makedirs(frames_path,
                exist_ok=True)

    for idx, t in tqdm(list(enumerate(np.linspace(0, 1, 120)))):
        curr_extra_embedding = tf.broadcast_to(tf.reshape((1 - t) * e1 + t * e2, (1, -1)),
                                                (curr_X_size.shape[0], e1.shape[0]))

        Y_pred_cat, Y_pred, _ = model([curr_X_glyph, curr_X_size, curr_X_pos, curr_X_font, curr_extra_embedding],
                                      training=True)

        Y_pred = ((1 - Y_pred.numpy().reshape((BITMAP_SIZE, BITMAP_SIZE))) * 255).astype(np.uint8)

        images.append(Y_pred)

        # imsave(f'/home/ubuntu/data/results/implicit_multiple_weights/roboto/{EXP_NAME}/weight_interpolation/{idx:06d}.png', Y_pred)


    # mean_weights = [image.mean() for image in images]

    # with open(f'/home/ubuntu/data/results/implicit_multiple_weights/roboto/{EXP_NAME}/weight_inter.json', 'w') as f:
    #     json.dump(mean_weights, f)

    video_path = f'/home/ubuntu/data/results/implicit_multiple_fonts/exp2/{EXP_NAME}/font_animation.mp4'
    with imageio.get_writer(video_path, fps=60) as writer:
        for img in images:
            writer.append_data(cv2.resize(img, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST))


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
    main()
