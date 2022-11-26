import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from glob import glob
import tensorflow as tf
import cv2
import pandas as pd
import json


GLYPH = 'g'
EXP_NAME = 'lowercase_g'
FONT = 'times_new_roman'

def load_img(img_path):
    return 1 - np.asarray(Image.open(img_path)) / 255


def load_dir(dir_path):
    return [
            load_img(img_path)
            for img_path in sorted(glob(os.path.join(dir_path, "*.png")),
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    ]


def load_glyph_nums(root):
    subdir = next(os.walk(root))[1][0]
    res = sorted(glob(os.path.join(root, subdir, "*.png")),
                 key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    return [int(os.path.splitext(os.path.basename(x))[0]) for x in res]


def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square((y_pred - y_true)), axis=-1)


def init_model(sizes):
    model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1024, activation='relu', input_shape=(x_shape, )),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(bitmap_size * bitmap_size, activation='sigmoid'),
        ])

    return model


def run_experiment(subdirs, glyph_nums, glyph_metadata, sizes,
                   base_names, modifier_names,
                   eps=1e-8, loss_patience=1, model=None, n_epochs=1):
    x_shape = len(base_names) + len(modifier_names) + len(subdirs)
    
    if model is None:
        model = init_model(sizes)

    all_combinations = [
        (os.path.join(subdir, f'{glyph_num}.png'),
         glyph_num,
         subdir_idx)
        for subdir_idx, subdir in enumerate(subdirs)
        for glyph_num in glyph_nums
    ]
    
    np.random.shuffle(all_combinations)

    learning_rates = [5 * 1e-4, 1e-5]
    curr_lr_idx = 0
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rates[curr_lr_idx])

    loss_history = []
    countdown = loss_patience
    batch_size = 32

    for epoch in range(1, n_epochs):
        if epoch % 50 == 0:
            curr_lr_idx = (curr_lr_idx + 1) % len(learning_rates)
            tf.keras.backend.set_value(optimizer.learning_rate, learning_rates[curr_lr_idx])

        batch_idx = np.arange(len(all_combinations))
        np.random.shuffle(batch_idx)
        batch_losses = []
        batch_pb = tqdm(range(0, len(all_combinations), batch_size), desc=f'Epoch {epoch}/{n_epochs}',
                        position=0, leave=True)
        for curr_batch_idx in batch_pb:
            curr_batch_indices = batch_idx[curr_batch_idx:curr_batch_idx + batch_size]
            curr_X, curr_M, curr_Y = load_batch([
                all_combinations[i]
                for i in curr_batch_indices
            ], len(subdirs), glyph_metadata, bitmap_size, base_names, modifier_names)
            with tf.GradientTape() as tape:
                pred = model(curr_X, training=True)
                loss_value = tf.reduce_mean(masked_mse(curr_Y, pred, curr_M))

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            batch_losses.append(float(loss_value))
            batch_pb.set_postfix({'loss': np.mean(batch_losses)})

        batch_pb.close()


        loss_history.append(np.mean(batch_losses))
        if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < eps:
            if countdown == 0:
                break
            else:
                countdown -= 1
        else:
            countdown = loss_patience

    return loss_history, model


def load_batch(batch, n_sizes, glyph_metadata, bitmap_size, base_names, modifier_names):
    X = []
    M = []
    Y = []
    for img_path, ordered_glyph_idx, size_idx in batch:
        y = load_img(img_path)
        y_resized, downscale_idx = make_larger(y, bitmap_size)
        m = make_mask(downscale_idx, bitmap_size)

        glyph_record = glyph_metadata[glyph_metadata['idx'] == ordered_glyph_idx]
        x_glyph_base = np.zeros(len(base_names))
        x_glyph_base[glyph_record['base_name_idx'].values[0]] = 1
        x_glyph_modifiers = np.zeros(len(modifier_names))
        glyph_modifiers = glyph_record['modifier_indices'].values[0]
        if len(glyph_modifiers) > 0:
            x_glyph_modifiers[glyph_modifiers] = 1

        x_size = np.zeros(n_sizes)
        x_size[size_idx] = 1
        x = np.concatenate([x_glyph_base, x_glyph_modifiers, x_size])
        X.append(x)
        Y.append(y_resized.ravel())
        M.append(m.ravel())

    X = np.stack(X)
    M = np.stack(M).astype(np.float32)
    Y = np.stack(Y)

    return X, M, Y


def main():
    # root = '/home/ubuntu/data/ground_truth/times_new_roman'
    # root = '/data/ground_truth/tahoma'
    # root = '/data/ground_truth/arial'
    root = f'/home/ubuntu/data/ground_truth/{FONT}'
    glyph_nums = [ord(GLYPH)]
    glyph_metadata = pd.read_csv(os.path.join(root, 'glyphs.csv'))
    glyph_metadata['modifier_indices'] = glyph_metadata['modifier_indices'].apply(json.loads)

    with open('/home/ubuntu/data/glyph_names.json', 'r') as f:
        glyph_names = json.load(f)
        base_names = glyph_names['base']
        modifier_names = glyph_names['modifiers']

    subdirs = next(os.walk(root))[1]
    subdirs.sort(key=lambda x: int(x.split('_')[0]))
    # bitmap_size = int(subdirs[-1].split('_')[0])
    sizes = sorted([int(s.split('_')[0]) for s in subdirs])
    
    subdirs = [os.path.join(root, s) for s in subdirs]

    hist, model = run_experiment(subdirs, glyph_nums, glyph_metadata, sizes,
                                 base_names, modifier_names,
                                 eps=1e-10, loss_patience=20, n_epochs=800)

    results_root = os.path.join('/home/ubuntu/data/results/memorization_mlp/', FONT, EXP_NAME)

    os.makedirs(results_root, exist_ok=True)
    with open(os.path.join(results_root, 'loss_hist.json'), 'w') as f:
        json.dump(hist, f)

    model.save(os.path.join(results_root, 'model'))


if __name__ == '__main__':
    main()
