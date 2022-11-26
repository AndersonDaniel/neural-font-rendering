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


def make_larger(img, new_size):
    # For now we assume that the image is square, but this can be easily changed 
    # by operating separately on rows and columns.
    assert img.shape[0] == img.shape[1]
    
    old_size = img.shape[0]
  
    # This method only makes the image larger, not smaller.
    assert old_size <= new_size

    # We will use simple row and column duplication. For each original index we 
    # first generate a list of how many times it will be duplicated. For example 
    # if the input image is 4x4, the list [1, 2, 1, 2] means that the output will
    # be 6x6, and that output will be created by duplicating the second and 
    # fourth row and column.
    # Note: the following is just one specific way to decide which indices to 
    # duplicate and by how much.
    dup_list = np.full(old_size, new_size // old_size)
    inds = np.linspace(start=0, stop=old_size-1, num=new_size % old_size)
    inds = np.floor(inds).astype('intp')
    dup_list[inds] += 1

    # Convert the list into a monotonic non-decreasing list of indices to sample 
    # from the input array. For example, [1, 2, 1, 2] will be converted to
    # [0, 1, 1, 2, 3, 3] which are the row/column indices we would use from the 
    # input array.
    ind_list = np.repeat(np.arange(old_size), dup_list)

    # Save a list of "important" indices, which will be preserved when downscaling 
    # back to the original size.
    downscale_ind = np.concatenate(([1], ind_list[1:] - ind_list[:-1]))
    downscale_ind = np.arange(new_size)[downscale_ind.astype(np.bool)]

    # Sample the input image according to ind_list.
    # Return the new image and the indices for downscaling.
    return img[np.ix_(ind_list, ind_list)], downscale_ind
  
def make_smaller(img, downscale_ind):
    # Since we did all the hard work in make_larger(), here we only need to sample
    # according to downscale_ind.
    return img[np.ix_(downscale_ind, downscale_ind)]


def make_mask(downscale_ind, mask_size):
    mask = np.zeros((mask_size, mask_size), dtype=int)
    mask[np.ix_(downscale_ind, downscale_ind)] = 1

    return mask


def masked_mse(y_true, y_pred, m):
    return tf.reduce_mean(tf.square((y_pred - y_true) * tf.convert_to_tensor(m)), axis=-1)


def run_experiment(subdirs, glyph_nums, glyph_metadata, bitmap_size,
                   base_names, modifier_names,
                   eps=1e-8, loss_patience=1, model=None, n_epochs=1):
    x_shape = len(base_names) + len(modifier_names) + len(subdirs)
    if model is None:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1024, activation='relu', input_shape=(x_shape, )),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(bitmap_size * bitmap_size, activation='sigmoid'),
        ])

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

        if epoch == n_epochs // 2:
            model.save('/data/training/v1/model_checkpoints/epoch_halfway')

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
    # root = '/data/ground_truth/times_new_roman'
    # root = '/data/ground_truth/tahoma'
    root = '/data/ground_truth/arial'
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

    hist, model = run_experiment(subdirs, glyph_nums, glyph_metadata, bitmap_size,
                                 base_names, modifier_names,
                                 eps=1e-10, loss_patience=20, n_epochs=800)

    os.makedirs('/data/training/v1', exist_ok=True)
    os.makedirs('/data/results/v1', exist_ok=True)
    with open('/data/training/v1/loss_hist_arial.json', 'w') as f:
        json.dump(hist, f)

    model.save('/data/training/v1/model_arial')


if __name__ == '__main__':
    main()
