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


POSITIONAL_ENCODINGS = (2 ** np.linspace(0, 12, 32)).tolist()


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
    return [119]


def make_model(n_glyphs):
    x_glyph = tf.keras.layers.Input((1,))
    x_size = tf.keras.layers.Input((2 * len(POSITIONAL_ENCODINGS) + 1,))
    x_pos = tf.keras.layers.Input((2 * (2 * len(POSITIONAL_ENCODINGS) + 1),))

    embedding_layer = tf.keras.layers.Embedding(n_glyphs, 1280, input_length=1)
    embedding = tf.keras.layers.Flatten()(embedding_layer(x_glyph))

    embedding_size_modification_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, input_shape=(embedding.shape[1] + x_size.shape[1],),
                              activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(embedding.shape[1], activation='tanh'),
    ])

    modified_embedding = embedding + embedding_size_modification_model(
        tf.keras.layers.Concatenate()([x_size, embedding])
    )
    
    x = tf.keras.layers.Concatenate()([x_pos, modified_embedding])
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = x + tf.keras.layers.Dense(256, activation='relu')(x)
    x = x + tf.keras.layers.Dense(256, activation='relu')(x)
    x = x + tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Concatenate()([x, modified_embedding])
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = x + tf.keras.layers.Dense(256, activation='relu')(x)
    x = x + tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = x + tf.keras.layers.Dense(256, activation='relu')(x)
    x = x + tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.models.Model(inputs=[x_glyph, x_size, x_pos], outputs=[x])


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
    n_first = int(n_epochs * .5)
    n_second = n_epochs - n_first

    return np.concatenate([
        make_lr_schedule(n_first, top=1e-4, lowest=1e-9),
        make_lr_schedule(n_second, top=1e-7, lowest=1e-11),
    ])


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
        model = make_model(glyph_metadata.shape[0])

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
    batch_size = 4
    loss_power_min = 1
    loss_power_max = 12

    for epoch in range(1, n_epochs):
        loss_power = loss_power_min + (loss_power_max - loss_power_min) * ((epoch - 1) / n_epochs)
        tf.keras.backend.set_value(optimizer.learning_rate, learning_rates[epoch - 1])

        # if epoch % 10 == 0:
            # model.save(f'/data/training/v6_small/model_checkpoints/checkpoint_{epoch}')

        batch_idx = np.arange(len(all_combinations))
        np.random.shuffle(batch_idx)
        batch_losses = []

        batch_pb = tqdm(range(0, len(all_combinations), batch_size), desc=f'Epoch {epoch}/{n_epochs}',
                        position=0, leave=True)
        for curr_batch_idx in batch_pb:
            curr_batch_indices = batch_idx[curr_batch_idx:curr_batch_idx + batch_size]
            curr_X_glyph, curr_X_size, curr_X_pos, curr_Y = load_batch([
                all_combinations[i]
                for i in curr_batch_indices
            ], len(subdirs), glyph_metadata, bitmap_size, base_names, modifier_names)

            with tf.GradientTape() as tape:
                pred = model([curr_X_glyph, curr_X_size, curr_X_pos], training=True)[:, 0]
                loss_value = tf.reduce_mean((pred - curr_Y) ** 2)
                real_loss_value = loss_value
                # real_loss_value = tf.reduce_mean(((pred - curr_Y) ** 2) ** loss_power) ** (1/loss_power)

            grads = tape.gradient(real_loss_value, model.trainable_weights)
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
        [np.sin(pe * np.pi * v), np.cos(pe * np.pi * v)]
        for pe in POSITIONAL_ENCODINGS
    ])


def full_evaluation_report(all_combinations, n_sizes, glyph_metadata,
                           bitmap_size, base_names, modifier_names, model):
    sample_pb = tqdm(all_combinations, desc=f'Evaluating', position=0, leave=True)
    losses = []
    max_loss = 0
    for sample in sample_pb:
        curr_X_glyph, curr_X_size, curr_X_pos, curr_Y = load_batch([sample], n_sizes, glyph_metadata,
                                    bitmap_size, base_names, modifier_names)
        pred_Y = model([curr_X_glyph, curr_X_size, curr_X_pos])[:, 0]
        curr_loss = float(tf.reduce_mean((curr_Y - pred_Y) ** 2))
        losses.append(curr_loss)
        max_loss = max(max_loss, curr_loss)
        sample_pb.set_postfix({'loss': np.mean(losses), 'max_loss': max_loss})

    return max_loss



def load_batch(batch, n_sizes, glyph_metadata, bitmap_size, base_names, modifier_names):
    X_glyph = []
    X_size = []
    X_pos = []
    Y = []
    for img_path, ordered_glyph_idx, size_idx in batch:
        y = load_img(img_path)

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
        
        Y += y.ravel().tolist()

    X_glyph = np.stack(X_glyph)
    X_size = np.stack(X_size)
    X_pos = np.stack(X_pos)
    Y = np.stack(Y)

    return X_glyph, X_size, X_pos, Y


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
    bitmap_size = int(subdirs[-1].split('_')[0])
    subdirs = [os.path.join(root, s) for s in subdirs]

    hist, max_hist, model = run_experiment(subdirs, glyph_nums, glyph_metadata, bitmap_size,
                                           base_names, modifier_names,
                                           eps=0, loss_patience=20, n_epochs=1000)

    os.makedirs('/data/training/v6_real_small/w', exist_ok=True)
    os.makedirs('/data/results/v6_real_small/w', exist_ok=True)
    with open('/data/training/v6_real_small/w/loss_hist_times_new_roman.json', 'w') as f:
        json.dump(hist, f)

    with open('/data/training/v6_real_small/w/loss_max_hist_times_new_roman.json', 'w') as f:
        json.dump(max_hist, f)

    model.save('/data/training/v6_real_small/w/model_times_new_roman')


if __name__ == '__main__':
    main()
