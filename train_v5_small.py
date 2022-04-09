import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from glob import glob
import tensorflow as tf
import cv2
import pandas as pd
import json
from tf_utils import ConvergenceEarlyStopping


POSITIONAL_ENCODINGS = [.1, .5, 1, 2, 10]


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

    # return [int(os.path.splitext(os.path.basename(x))[0]) for x in res]
    return [71]


def run_experiment(subdirs, glyph_nums, glyph_metadata, bitmap_size,
                   base_names, modifier_names,
                   eps=1e-8, loss_patience=1, model=None, n_epochs=1):
    x_shape = (glyph_metadata.shape[0]  # Glyphs
            #    + len(modifier_names)  # Modifiers
               + 3 *  # Positional encodings for size / vert / hori positions
                     (1  # value itself
                      + 2 * len(POSITIONAL_ENCODINGS)  # sin/cos encodings
                ))

    if model is None:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(2048, activation='relu', input_shape=(x_shape, )),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])

    all_combinations = [
        (os.path.join(subdir, f'{glyph_num}.png'),
         glyph_num,
         subdir_idx)
        for subdir_idx, subdir in enumerate(subdirs)
        for glyph_num in glyph_nums
    ]
    
    np.random.shuffle(all_combinations)

    learning_rates = [5 * 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]
    curr_lr_idx = 0
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rates[curr_lr_idx])

    loss_history = []
    max_loss_history = []
    countdown = loss_patience
    batch_size = 4

    for epoch in range(1, n_epochs):
        if epoch % 5 == 0:
            curr_lr_idx = (curr_lr_idx + 1) % len(learning_rates)
            tf.keras.backend.set_value(optimizer.learning_rate, learning_rates[curr_lr_idx])

        # if epoch % 10 == 0:
            # model.save(f'/data/training/v5/model_checkpoints/checkpoint_{epoch}')

        batch_idx = np.arange(len(all_combinations))
        np.random.shuffle(batch_idx)
        batch_losses = []

        batch_pb = tqdm(range(0, len(all_combinations), batch_size), desc=f'Epoch {epoch}/{n_epochs}',
                        position=0, leave=True)
        for curr_batch_idx in batch_pb:
            curr_batch_indices = batch_idx[curr_batch_idx:curr_batch_idx + batch_size]
            curr_X, curr_Y = load_batch([
                all_combinations[i]
                for i in curr_batch_indices
            ], len(subdirs), glyph_metadata, bitmap_size, base_names, modifier_names)

            with tf.GradientTape() as tape:
                pred = model(curr_X, training=True)[:, 0]
                loss_value = tf.reduce_mean((pred - curr_Y) ** 2)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            batch_losses.append(float(loss_value))
            batch_pb.set_postfix({'loss': np.mean(batch_losses)})

        batch_pb.close()

        loss_history.append(np.mean(batch_losses))

        if epoch % 5 == 0:
            max_loss = full_evaluation_report(all_combinations, len(subdirs), glyph_metadata,
                                              bitmap_size, base_names, modifier_names, model)
            max_loss_history.append(max_loss)

        if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < eps:
            if countdown == 0:
                break
            else:
                countdown -= 1
        else:
            countdown = loss_patience
        
    return loss_history, max_loss_history, model


def positional_encodings(v):
    return np.concatenate([
        [np.sin(pe * v), np.cos(pe * v)]
        for pe in POSITIONAL_ENCODINGS
    ])


def full_evaluation_report(all_combinations, n_sizes, glyph_metadata,
                           bitmap_size, base_names, modifier_names, model):
    sample_pb = tqdm(all_combinations, desc=f'Evaluating', position=0, leave=True)
    losses = []
    max_loss = 0
    for sample in sample_pb:
        curr_X, curr_Y = load_batch([sample], n_sizes, glyph_metadata,
                                    bitmap_size, base_names, modifier_names)
        pred_Y = model(curr_X)[:, 0]
        curr_loss = float(tf.reduce_mean((curr_Y - pred_Y) ** 2))
        losses.append(curr_loss)
        max_loss = max(max_loss, curr_loss)
        sample_pb.set_postfix({'loss': np.mean(losses), 'max_loss': max_loss})

    return max_loss



def load_batch(batch, n_sizes, glyph_metadata, bitmap_size, base_names, modifier_names):
    X = []
    Y = []
    for img_path, ordered_glyph_idx, size_idx in batch:
        y = load_img(img_path)

        glyph_record = glyph_metadata[glyph_metadata['idx'] == ordered_glyph_idx]
        x_glyph_base = np.zeros(glyph_metadata.shape[0])
        x_glyph_base[glyph_record.index.values[0]] = 1

        relative_size = size_idx / n_sizes
        x_base = np.concatenate([x_glyph_base,
                                 [relative_size],
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
                x_curr = np.concatenate([x_base.copy(),
                                        [vert_pos, hori_pos],
                                        vert_encodings, hori_encodings])
                X.append(x_curr)
        
        Y += y.ravel().tolist()

    X = np.stack(X)
    Y = np.stack(Y)

    return X, Y


def main():
    root = '/data/ground_truth/times_new_roman'
    glyph_nums = load_glyph_nums(root)
    glyph_metadata = pd.read_csv(os.path.join(root, 'glyphs.csv'))
    glyph_metadata['modifier_indices'] = glyph_metadata['modifier_indices'].apply(json.loads)

    with open('/data/glyph_names.json', 'r') as f:
        glyph_names = json.load(f)
        base_names = glyph_names['base']
        modifier_names = glyph_names['modifiers']

    subdirs = next(os.walk(root))[1]
    subdirs.sort(key=lambda x: int(x.split('_')[0]))
    subdirs = [subdirs[3], subdirs[len(subdirs) // 2], subdirs[-4]]
    bitmap_size = int(subdirs[-1].split('_')[0])
    subdirs = [os.path.join(root, s) for s in subdirs]

    hist, max_hist, model = run_experiment(subdirs, glyph_nums, glyph_metadata, bitmap_size,
                                           base_names, modifier_names,
                                           eps=0, loss_patience=20, n_epochs=10000)

    os.makedirs('/data/training/v5_real_small_generalization/cap_g', exist_ok=True)
    os.makedirs('/data/results/v5_real_small_generalization/cap_g', exist_ok=True)
    with open('/data/training/v5_real_small_generalization/cap_g/loss_hist_times_new_roman.json', 'w') as f:
        json.dump(hist, f)

    with open('/data/training/v5_real_small_generalization/cap_g/loss_max_hist_times_new_roman.json', 'w') as f:
        json.dump(max_hist, f)

    model.save('/data/training/v5_real_small_generalization/cap_g/model_times_new_roman')


if __name__ == '__main__':
    main()
