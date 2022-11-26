import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from glob import glob
import tensorflow as tf
import cv2
import pandas as pd
import json
import gc
from time import time


# FONT = 'times_new_roman'
# EXP_NAME = 'lowercase_g'


INTENSITY_CATEGORIES = 21
FOCAL_LOSS_GAMMA = 6


def focal_loss_cce(y, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)
    return -tf.reduce_mean(
        tf.reduce_sum(
            y * ((1 - y_pred) ** FOCAL_LOSS_GAMMA) * tf.math.log(y_pred),
            axis=1
        )
    )


def cce(y, y_pred):
    return -tf.reduce_mean(
        tf.reduce_sum(
            y * tf.math.log(tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)),
            axis=1
        )
    )


def load_img(img_path):
    return 1 - np.asarray(Image.open(img_path)) / 255


def load_dir(dir_path):
    return [
            load_img(img_path)
            for img_path in sorted(glob(os.path.join(dir_path, "*.png")),
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    ]


# def load_glyph_nums(root):
#     subdir = next(os.walk(root))[1][0]
#     res = sorted(glob(os.path.join(root, subdir, "*.png")),
#                  key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

#     # return [int(os.path.splitext(os.path.basename(x))[0]) for x in res]
#     return [103]


def make_model(n_glyphs):
    x_glyph = tf.keras.layers.Input((1,))
    x_pos = tf.keras.layers.Input((2,))

    # embedding_layer = tf.keras.layers.Embedding(n_glyphs, 1280, input_length=1)
    # embedding = tf.keras.layers.Flatten()(embedding_layer(x_glyph))
    
    # x = tf.keras.layers.Concatenate()([x_pos, embedding])
    x = x_pos

    x = tf.keras.layers.Dense(784, activation='relu')(x)
    x = tf.keras.layers.Dense(784, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = x + tf.keras.layers.Dense(784, activation='relu')(x)
    x = x + tf.keras.layers.Dense(784, activation='relu')(x)
    x = x + tf.keras.layers.Dense(784, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Concatenate()([x, embedding])
    x = tf.keras.layers.Dense(784, activation='relu')(x)
    x = x + tf.keras.layers.Dense(784, activation='relu')(x)
    x = x + tf.keras.layers.Dense(784, activation='relu')(x)
    x = x + tf.keras.layers.Dense(784, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = x + tf.keras.layers.Dense(784, activation='relu')(x)
    x = x + tf.keras.layers.Dense(784, activation='relu')(x)
    x = tf.keras.layers.Dense(INTENSITY_CATEGORIES, activation='softmax')(x)

    return tf.keras.models.Model(inputs=[x_glyph, x_pos], outputs=[x])


# def make_lr_schedule(n_epochs, initial=1e-4, top=5 * 1e-4, lowest=5 * 1e-6):
#   warmup = n_epochs // 10
#   exponential_decay = n_epochs - warmup
#   decay_factor = (lowest / top) ** (1 / exponential_decay)
#   return np.concatenate([initial + (top - initial) * np.arange(warmup),
#                          top * decay_factor ** np.arange(exponential_decay)])


def make_lr_schedule(n_epochs, top=1e-4, lowest=1e-9):
    exponential_decay = n_epochs
    decay_factor = (lowest / top) ** (1 / exponential_decay)
    return top * decay_factor ** np.arange(exponential_decay)


def make_piecewise_lr_schedule(n_epochs):
    n_first = n_second = n_third = n_fourth = int(n_epochs / 5)
    n_fifth = n_epochs - n_first - n_second - n_third - n_fourth

    return np.concatenate([
        make_lr_schedule(n_first, top=1e-3, lowest=1e-5),
        make_lr_schedule(n_second, top=1e-4, lowest=1e-6),
        make_lr_schedule(n_third, top=1e-4 / 2, lowest=1e-7),
        make_lr_schedule(n_fourth, top=1e-5, lowest=1e-8),
        make_lr_schedule(n_fifth, top=1e-6, lowest=1e-9),
    ])


def run_experiment(subdirs, glyph_nums, glyph_metadata, bitmap_size,
                   base_names, modifier_names,
                   eps=1e-8, loss_patience=1, model=None, n_epochs=1):
    x_shape = (glyph_metadata.shape[0]  # Glyphs
            #    + len(modifier_names)  # Modifiers
               + 3 *  # Positional encodings for size / vert / hori positions
                     (1  # value itself
                ))

    if model is None:
        model = make_model(glyph_metadata.shape[0])

    # model.summary()
    # exit()

    all_combinations = [
        (os.path.join(subdir, f'{glyph_num}.png'),
         glyph_num,
         subdir_idx)
        for subdir_idx, subdir in enumerate(subdirs)
        for glyph_num in glyph_nums
    ]
    
    np.random.shuffle(all_combinations)

    # learning_rates = make_lr_schedule(n_epochs)
    learning_rates = make_piecewise_lr_schedule(n_epochs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rates[0])
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    loss_history = []
    max_loss_history = []
    countdown = loss_patience
    batch_size = 1
    loss_power = 3

    for epoch in range(1, n_epochs):
        tf.keras.backend.set_value(optimizer.learning_rate, learning_rates[epoch - 1])

        batch_idx = np.arange(len(all_combinations))
        np.random.shuffle(batch_idx)
        batch_losses = []

        load_batch_times = []
        forward_times = []
        backward_times = []
        batch_pb = tqdm(range(0, len(all_combinations), batch_size), desc=f'Epoch {epoch}/{n_epochs}',
                        position=0, leave=True)
        for curr_batch_idx in batch_pb:
            t1 = time()

            curr_batch_indices = batch_idx[curr_batch_idx:curr_batch_idx + batch_size]
            curr_X_glyph, curr_X_pos, curr_Y, _ = load_batch([
                all_combinations[i]
                for i in curr_batch_indices
            ], len(subdirs), glyph_metadata, bitmap_size, base_names, modifier_names)

            t2 = time()

            with tf.GradientTape() as tape:
                pred = model([curr_X_glyph, curr_X_pos], training=True)
                # loss_value = cce(curr_Y, pred)
                loss_value = focal_loss_cce(curr_Y, pred)
                # real_loss_value = tf.reduce_mean(((pred - curr_Y) ** 2) ** loss_power) ** (1/loss_power)

            t3 = time()

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            batch_losses.append(float(loss_value))

            t4 = time()

            load_batch_times.append(t2 - t1)
            forward_times.append(t3 - t2)
            backward_times.append(t4 - t3)

            batch_pb.set_postfix({'loss': np.mean(batch_losses),
                                  'load_batch_time': np.mean(load_batch_times),
                                  'forward_time': np.mean(forward_times),
                                  'backward_time': np.mean(backward_times)})

        batch_pb.close()

        loss_history.append(np.mean(batch_losses))

        # if epoch % 5 == 0:
        #     max_loss = full_evaluation_report(all_combinations, len(subdirs), glyph_metadata,
        #                                       bitmap_size, base_names, modifier_names, model)
        #     max_loss_history.append(max_loss)

        if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < eps:
            if countdown == 0:
                break
            else:
                countdown -= 1
        else:
            countdown = loss_patience
        
    return loss_history, max_loss_history, model


def load_batch(batch, n_sizes, glyph_metadata, bitmap_size, base_names, modifier_names):
    X_glyph = []
    X_pos = []
    Y = []
    for img_path, ordered_glyph_idx, size_idx in batch:
        y = load_img(img_path)

        glyph_record = glyph_metadata[glyph_metadata['idx'] == ordered_glyph_idx]
        x_glyph_base = [glyph_record.index.values[0]]

        relative_size = size_idx / n_sizes
        
        height = y.shape[0]
        width = y.shape[1]
        for row in range(height):
            vert_pos = -1 + (2 * row + 1) / height
            for col in range(width):
                hori_pos = -1 + (2 * col + 1) / width
                x_pos = np.array([vert_pos, hori_pos])
                
                X_glyph.append(x_glyph_base)
                X_pos.append(x_pos)
        
        Y += y.ravel().tolist()

    X_glyph = np.stack(X_glyph)
    X_pos = np.stack(X_pos)
    Y = np.stack(Y)

    categorical_Y = np.zeros((Y.shape[0], INTENSITY_CATEGORIES))
    categorical_Y[range(Y.shape[0]), (np.floor(Y / (1 / (INTENSITY_CATEGORIES - 1)))).astype(int)] = 1

    return X_glyph, X_pos, categorical_Y, Y


def main():
    # for FONT in ['times_new_roman', 'arial', 'tahoma']:
    for FONT in ['times_new_roman']:
        for glyph_idx in range(ord('a'), ord('z') + 1):
        # for glyph_idx in [*range(ord('A'), ord('Z') + 1), *range(ord('0'), ord('9') + 1),
                        #   *',./?><!@#$%^&*()-=_+']:
        # for glyph_idx in map(ord, list(',./?><!@#$%^&*()-=_+')):
            glyph_nums = [glyph_idx]
            GLYPH = chr(glyph_idx)
            EXP_NAME = f'lowercase_{GLYPH}'
        #     EXP_NAME = str(glyph_idx)

            root = f'/home/ubuntu/data/ground_truth/{FONT}'
            # glyph_nums = load_glyph_nums(root)
            glyph_metadata = pd.read_csv(os.path.join(root, 'glyphs.csv'))
            glyph_metadata['modifier_indices'] = glyph_metadata['modifier_indices'].apply(json.loads)

            with open('/home/ubuntu/data/glyph_names.json', 'r') as f:
                glyph_names = json.load(f)
                base_names = glyph_names['base']
                modifier_names = glyph_names['modifiers']

            subdirs = next(os.walk(root))[1]
            subdirs.sort(key=lambda x: int(x.split('_')[0]))
            bitmap_size = int(subdirs[-1].split('_')[0])
            subdirs = [os.path.join(root, s) for s in subdirs]

            # print(subdirs)
            # exit()

            hist, max_hist, model = run_experiment(subdirs, glyph_nums, glyph_metadata, bitmap_size,
                                                base_names, modifier_names,
                                                eps=0, loss_patience=20, n_epochs=500)

            results_root = os.path.join('/home/ubuntu/data/results/implicit_no_freq_encoding/', FONT, EXP_NAME)
            os.makedirs(os.path.join(results_root), exist_ok=True)
            with open(os.path.join(results_root, 'loss_hist.json'), 'w') as f:
                json.dump(hist, f)

            # with open('/data/training/v8_real_small/abcxyz/loss_max_hist_times_new_roman.json', 'w') as f:
            #     json.dump(max_hist, f)

            model.save(os.path.join(results_root, 'model'))


if __name__ == '__main__':
    main()
