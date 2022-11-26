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
import matplotlib.pyplot as plt
import imageio

# import tracemalloc
# tracemalloc.start(20)


POSITIONAL_ENCODINGS = (2 ** np.linspace(0, 12, 32)).tolist()
INTENSITY_CATEGORIES = 21
REG_LAMBDA = 0
FOCAL_LOSS_GAMMA = 6
# GLYPH_NUMS = [ord('g')]
# EXP_NAME = f'lower_case_{chr(GLYPH_NUMS[0])}'
GLYPH_NUMS = [ord('1')]
EXP_NAME = 'one'

def cce(y, y_pred, w):
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)
    return -tf.tensordot(
        tf.reduce_sum(
            y * tf.math.log(y_pred),
            axis=1
        ), w, 1
    ) / tf.reduce_sum(w)


def focal_loss_cce(y, y_pred, w):
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)
    return -tf.tensordot(
        tf.reduce_sum(
            y * ((1 - y_pred) ** FOCAL_LOSS_GAMMA) * tf.math.log(y_pred),
            axis=1
        ), w, 1
    ) / tf.reduce_sum(w)


def load_img(img_path):
    return 1 - np.asarray(Image.open(img_path)) / 255


def load_dir(dir_path):
    return [
            load_img(img_path)
            for img_path in sorted(glob(os.path.join(dir_path, "*.png")),
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    ]


def make_model(n_glyphs, n_fonts):
    x_glyph = tf.keras.layers.Input((1,))
    x_size = tf.keras.layers.Input((2 * len(POSITIONAL_ENCODINGS) + 1,))
    x_pos = tf.keras.layers.Input((2 * (2 * len(POSITIONAL_ENCODINGS) + 1),))
    x_font = tf.keras.layers.Input((1,))
    x_extra = tf.keras.layers.Input((1,))
    w = tf.convert_to_tensor(np.linspace(0, 1, INTENSITY_CATEGORIES).reshape((-1, 1)).astype(np.float32))

    EMBEDDING_DIM = 1420

    # font_embedding_layer = tf.keras.layers.Embedding(n_fonts, EMBEDDING_DIM, input_length=1)
    # font_embedding = tf.keras.layers.Flatten()(font_embedding_layer(x_font))

    embedding_extra_modification_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(300, input_shape=(x_extra.shape[1],),
                              activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(EMBEDDING_DIM, activation='tanh')
    ])

    # embedding_extra_modification_model = tf.keras.models.Sequential([
    #     tf.keras.layers.Dense(300, input_shape=(x_extra.shape[1] + font_embedding.shape[1],),
    #                           activation='relu'),
    #     tf.keras.layers.Dense(300, activation='relu'),
    #     tf.keras.layers.Dense(300, activation='relu'),
    #     tf.keras.layers.Dense(300, activation='relu'),
    #     tf.keras.layers.Dense(300, activation='relu'),
    #     tf.keras.layers.Dense(EMBEDDING_DIM, activation='tanh')
    # ])

    # modified_embedding = font_embedding + embedding_extra_modification_model(
    #     tf.keras.layers.Concatenate()([x_extra, font_embedding])
    # )

    # modified_embedding = embedding_extra_modification_model(x_extra)

    o_modified_embedding = tf.keras.layers.Input((EMBEDDING_DIM,))

    embedding_size_modification_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(300, input_shape=(EMBEDDING_DIM + x_size.shape[1],),
                              activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(EMBEDDING_DIM, activation='tanh'),
    ])

    # modified_embedding = modified_embedding + embedding_size_modification_model(
    #     tf.keras.layers.Concatenate()([x_size, modified_embedding])
    # )

    modified_embedding = o_modified_embedding + embedding_size_modification_model(
        tf.keras.layers.Concatenate()([x_size, o_modified_embedding])
    )
    
    x = tf.keras.layers.Concatenate()([x_pos, modified_embedding])
    x = tf.keras.layers.Dense(1560, activation='relu')(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = x + tf.keras.layers.Dense(1024, activation='relu')(x)
    x = x + tf.keras.layers.Dense(1024, activation='relu')(x)
    x = x + tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Concatenate()([x, modified_embedding])
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = x + tf.keras.layers.Dense(1024, activation='relu')(x)
    x = x + tf.keras.layers.Dense(1024, activation='relu')(x)
    x = x + tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = x + tf.keras.layers.Dense(1024, activation='relu')(x)
    x = x + tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(INTENSITY_CATEGORIES, activation='softmax')(x)
    y = tf.matmul(x, w)[:, 0]
    
    # return tf.keras.models.Model(inputs=[x_glyph, x_size, x_pos, x_font, x_extra], outputs=[x, y, modified_embedding])
    return (embedding_extra_modification_model,
            tf.keras.models.Model(inputs=[x_glyph, x_size, x_pos, x_font, o_modified_embedding],
                                  outputs=[x, y, modified_embedding]))


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


def run_experiment(subdirs, glyph_nums, glyph_metadata, bitmap_size, n_sizes,
                   base_names, modifier_names,
                   eps=1e-8, loss_patience=1, model=None, n_epochs=1):
    if model is None:
        extra_embedding_model, model = make_model(glyph_metadata.shape[0], len(subdirs))

    all_combinations = [
        (os.path.join(subdir, f'{glyph_num}.png'),
         glyph_num,
         subdir_idx, font_idx, e)
        for font_idx, (font, font_subdirs) in enumerate(subdirs.items())
        for e, e_subdirs in font_subdirs.items()
        for subdir_idx, subdir in enumerate(e_subdirs)
        for glyph_num in glyph_nums
    ]
    
    np.random.shuffle(all_combinations)

    # learning_rates = make_lr_schedule(n_epochs)
    learning_rates = make_piecewise_lr_schedule(n_epochs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rates[0])
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    loss_history = []
    max_loss_history = []
    visualization_history = []
    countdown = loss_patience
    batch_size = 1

    mse_weight = 0
    loss_power = 3

    # n_epochs = 6
    # snapshot1 = tracemalloc.take_snapshot()

    for epoch in range(1, n_epochs):
        if get_free_ram_percent() <= .075:
            print('Breaking due to low RAM!')
            break

        tf.keras.backend.set_value(optimizer.learning_rate, learning_rates[epoch - 1])

        batch_idx = np.arange(len(all_combinations))
        np.random.shuffle(batch_idx)
        batch_losses = []
        batch_bce_losses = []
        batch_mse_losses = []

        load_batch_times = []
        forward_times = []
        backward_times = []
        batch_pb = tqdm(range(0, len(all_combinations), batch_size), desc=f'Epoch {epoch}/{n_epochs}',
                        position=0, leave=True)
        for curr_batch_idx in batch_pb:
            t1 = time()

            curr_batch_indices = batch_idx[curr_batch_idx:curr_batch_idx + batch_size]
            curr_X_glyph, curr_X_size, curr_X_pos, curr_X_font, curr_X_extra, curr_Y_cat, curr_Y, W = load_batch([
                all_combinations[i]
                for i in curr_batch_indices
            ], n_sizes, glyph_metadata, bitmap_size, base_names, modifier_names)

            t2 = time()

            with tf.GradientTape() as tape:
                extra_embedding = extra_embedding_model(curr_X_extra, training=True)
                # pred_cat, pred, embedding  = model([curr_X_glyph, curr_X_size, curr_X_pos, curr_X_font, curr_X_extra], training=True)
                pred_cat, pred, embedding  = model([curr_X_glyph, curr_X_size, curr_X_pos, curr_X_font, extra_embedding],
                                                   training=True)
                # bce_loss_value = cce(curr_Y_cat, pred_cat, W) + REG_LAMBDA * tf.reduce_mean(tf.reduce_sum(embedding ** 2, axis=1))
                bce_loss_value = focal_loss_cce(curr_Y_cat, pred_cat, W) + REG_LAMBDA * tf.reduce_mean(tf.reduce_sum(embedding ** 2, axis=1))
                mse_loss_value = tf.reduce_mean(((curr_Y - pred) ** 2) ** loss_power) ** (1 / loss_power)
                loss_value = bce_loss_value + mse_weight * mse_loss_value
                # real_loss_value = tf.reduce_mean(((pred - curr_Y) ** 2) ** loss_power) ** (1/loss_power)

            t3 = time()

            grads = tape.gradient(loss_value, extra_embedding_model.trainable_weights + model.trainable_weights)
            optimizer.apply_gradients(zip(grads, extra_embedding_model.trainable_weights + model.trainable_weights))
            batch_losses.append(float(loss_value))
            batch_bce_losses.append(float(bce_loss_value))
            batch_mse_losses.append(float(mse_loss_value))

            t4 = time()

            load_batch_times.append(t2 - t1)
            forward_times.append(t3 - t2)
            backward_times.append(t4 - t3)

            batch_pb.set_postfix({'loss': np.mean(batch_losses),
                                #   'load_batch_time': np.mean(load_batch_times).round(3),
                                #   'forward_time': np.mean(forward_times).round(3),
                                #   'backward_time': np.mean(backward_times).round(3),
                                #   'free_mem_pct': get_free_ram_percent(),
                                'bce_loss': np.mean(batch_bce_losses),
                                'mse_loss': np.mean(batch_mse_losses)
                                })

            # break

        batch_pb.close()

        loss_history.append(np.mean(batch_losses))

        # if epoch % 5 == 0 or epoch + 1 == n_epochs:
        #     visualization_history.append(visualize_epoch(all_combinations, len(subdirs), glyph_metadata,
        #                                                  bitmap_size, base_names,
        #                                                  modifier_names, model))

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

    # snapshot2 = tracemalloc.take_snapshot()

    # top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    # top_stats = snapshot2.statistics('traceback')

    # print('[ Top 10 differences ]')
    # for stat in top_stats[:10]:
    #     print(stat)

    # stat = top_stats[0]
    # print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
    # for line in stat.traceback.format():
    #     print(line)
        
    return loss_history, max_loss_history, extra_embedding_model, model, visualization_history


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
        curr_X_glyph, curr_X_size, curr_X_pos, curr_X_font, curr_X_extra, curr_Y_cat, curr_Y, W = load_batch([sample], n_sizes, glyph_metadata,
                                    bitmap_size, base_names, modifier_names)
        pred_Y_cat, pred_Y, _ = model([curr_X_glyph, curr_X_size, curr_X_pos, curr_X_font, curr_X_extra])
        curr_loss = float(cce(curr_Y_cat, pred_Y_cat, W))
        losses.append(curr_loss)
        max_loss = max(max_loss, curr_loss)
        sample_pb.set_postfix({'loss': np.mean(losses), 'max_loss': max_loss})

    return max_loss


def visualize_epoch(all_combinations, n_sizes, glyph_metadata,
                    bitmap_size, base_names, modifier_names, model):
    all_images = []
    all_fonts = []
    all_extras = []
    all_size_indices = []
    unique_fonts = set()
    unique_extras = set()
    sizes = []
    for comb in tqdm(all_combinations):
        curr_X_glyph, curr_X_size, curr_X_pos, curr_X_font, curr_X_extra, curr_Y_cat, curr_Y, W = load_batch([comb],
                n_sizes, glyph_metadata, bitmap_size, base_names, modifier_names)

        pred_cat, pred, embedding = model([curr_X_glyph, curr_X_size, curr_X_pos, curr_X_font, curr_X_extra], training=True)
        Y_pred = pred.numpy()
        size = int(np.sqrt(Y_pred.shape[0]))
        img = Y_pred.reshape((-1, size))

        sizes.append(size)
        all_size_indices.append(comb[2])
        all_fonts.append(curr_X_font[0, 0])
        all_extras.append(curr_X_extra[0, 0])
        unique_fonts.add(all_fonts[-1])
        unique_extras.add(all_extras[-1])

        img = ((1 - img) * 255).astype(np.uint8)

        err_img, gt_img, curr_error = get_error_image(Y_pred, curr_Y)
        
        res_img = h_concat_images(img, err_img, gt_img)
        all_images.append(res_img)

    largest_size = max(sizes)
    for i in range(len(all_images)):
        img = all_images[i]
        size = sizes[i]
        padding = (largest_size - size) // 2
        img_full = np.ones((3 * largest_size, largest_size, 3), dtype=img.dtype) * 255
        img_full[padding:padding + size, padding:padding + size] = img[:, :size]
        img_full[padding + size:padding + 2 * size, padding:padding + size] = img[:, size:2 * size]
        img_full[padding + 2 * size:padding + 3 * size, padding:padding + size] = \
            img[:, 2 * size:3 * size]

        all_images[i] = img_full

    sorted_indices = (np.array(all_size_indices) * 1000 + np.array(all_extras)).argsort()
    all_images = [all_images[i] for i in sorted_indices]
    M = len(unique_extras)
    res = np.concatenate([
        np.concatenate(all_images[M * i:M * (i + 1)], axis=1)
        for i in range(len(all_images) // M)
    ], axis=0)

    return res
    

def get_error_image(y_pred, y):
    y_pred_expanded = y_pred.reshape((-1, int(np.sqrt(y_pred.shape[0]))))
    y_expanded = y.reshape((-1, int(np.sqrt(y.shape[0]))))
    diff = y_pred_expanded - y_expanded
    cmap = plt.get_cmap('RdBu')
    img = cmap((diff + 1) / 2)[..., :-1]
    return (np.uint8(img * 255),
            ((1 - y_expanded) * 255).astype(np.uint8),
            (diff ** 2).mean())


def h_concat_images(*imgs):
    imgs = list(imgs)
    for i in range(len(imgs)):
        if len(imgs[i].shape) == 2:
            imgs[i] = np.stack(3 * [imgs[i]], axis=-1)

    return np.concatenate(imgs, axis=1)


def load_batch(batch, n_sizes, glyph_metadata, bitmap_size, base_names, modifier_names):
    X_glyph = []
    X_size = []
    X_pos = []
    X_font = []
    X_extra = []
    Y = []
    W = []
    for img_path, ordered_glyph_idx, size_idx, font_idx, extra in batch:
        y = load_img(img_path)
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


def get_free_ram_percent():
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t -m').readlines()[-1].split()[1:]
    )

    return free_memory / total_memory


def main():
    roots = {
        'exp2': {
            # 1/7: '/home/ubuntu/data/ground_truth/roboto/roboto-thin',
            # 2/7: '/home/ubuntu/data/ground_truth/roboto/roboto-light',
            # 3/7: '/home/ubuntu/data/ground_truth/roboto/roboto-regular',
            # 4/7: '/home/ubuntu/data/ground_truth/roboto/roboto-medium',
            # 6/7: '/home/ubuntu/data/ground_truth/roboto/roboto-bold',
            # -1: '/home/ubuntu/data/ground_truth/arial',
            # 1: '/home/ubuntu/data/ground_truth/times_new_roman'
            -1: '/home/ubuntu/data/ground_truth/lucida_sans',
            1: '/home/ubuntu/data/ground_truth/lucida_bright'
        }
    }

    glyph_nums = GLYPH_NUMS
    sample_dir = list(list(roots.values())[0].values())[0]
    glyph_metadata = pd.read_csv(os.path.join(*os.path.split(sample_dir), 'glyphs.csv'))
    glyph_metadata['modifier_indices'] = glyph_metadata['modifier_indices'].apply(json.loads)

    with open('/home/ubuntu/data/glyph_names.json', 'r') as f:
        glyph_names = json.load(f)
        base_names = glyph_names['base']
        modifier_names = glyph_names['modifiers']

    subdirs = {
        font: {
            k: next(os.walk(v))[1]
            for k, v in font_roots.items()
        }
        for font, font_roots in roots.items()
    }

    n_sizes = 0
    for font, font_subdirs in subdirs.items():
        for k, v in font_subdirs.items():
            v.sort(key=lambda x: int(x.split('_')[0]))
            n_sizes = len(v)

    bitmap_size = int(list(list(subdirs.values())[0].values())[0][-1].split('_')[0])
    subdirs = {
        font: {
            k: [os.path.join(roots[font][k], s) for s in v]
            for k, v in font_subdirs.items()
        }
        for font, font_subdirs in subdirs.items()
    }

    hist, max_hist, extra_embedding_model, model, visualization_history = run_experiment(subdirs, glyph_nums, glyph_metadata, bitmap_size, n_sizes,
                                                                  base_names, modifier_names,
                                                                  eps=0, loss_patience=20, n_epochs=500)


    results_root = os.path.join('/home/ubuntu/data/results/implicit_multiple_fonts', 'exp2', EXP_NAME)
    os.makedirs(os.path.join(results_root), exist_ok=True)

    with open(os.path.join(results_root, 'loss_hist.json'), 'w') as f:
        json.dump(hist, f)

    with open(os.path.join(results_root, 'loss_max_hist.json'), 'w') as f:
        json.dump(max_hist, f)

    model.save(os.path.join(results_root, 'model'))
    extra_embedding_model.save(os.path.join(results_root, 'extra_embedding_model'))


if __name__ == '__main__':
    main()
