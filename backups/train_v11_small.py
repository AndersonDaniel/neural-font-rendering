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


POSITIONAL_ENCODINGS = (2 ** np.linspace(0, 12, 32)).tolist()
INTENSITY_CATEGORIES = 21
REG_LAMBDA = 1e-3
FOCAL_LOSS_GAMMA = 6
GLYPH_NUMS = [ord('A')]
FONT_EMBEDDING_DIM = 1420
EMBEDDING_LOSS_WEIGHT = .005
DISCRIMINATOR_LOSS_WEIGHT = .025
# EMBEDDING_LOSS_WEIGHT = 1e-20
# DISCRIMINATOR_LOSS_WEIGHT = 1e-20
# EMBEDDING_LOSS_WEIGHT = 0
# DISCRIMINATOR_LOSS_WEIGHT = 0

FAKE_OPTIMIZER_LR_RATIO = 50


def focal_loss_cce(y, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)
    return -tf.reduce_mean(
        tf.reduce_sum(
            y * ((1 - y_pred) ** FOCAL_LOSS_GAMMA) * tf.math.log(y_pred),
            axis=1
        )
    )


def bce(y, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)
    return -tf.reduce_mean(
        y * tf.math.log(y_pred) + (1 - y) * tf.math.log(1 - y_pred)
    )


def load_img(img_path):
    return 1 - np.asarray(Image.open(img_path)) / 255


def load_dir(dir_path):
    return [
            load_img(img_path)
            for img_path in sorted(glob(os.path.join(dir_path, "*.png")),
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    ]


def make_inner_model():
    x_font = tf.keras.layers.Input((FONT_EMBEDDING_DIM,))
    x_size = tf.keras.layers.Input((2 * len(POSITIONAL_ENCODINGS) + 1,))
    x_pos = tf.keras.layers.Input((2 * (2 * len(POSITIONAL_ENCODINGS) + 1),))
    x_extra = tf.keras.layers.Input((1,))
    w = tf.convert_to_tensor(np.linspace(0, 1, INTENSITY_CATEGORIES).reshape((-1, 1)).astype(np.float32))

    embedding_extra_modification_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(300, input_shape=(x_font.shape[1] + x_extra.shape[1],),
                              activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(x_font.shape[1], activation='tanh')
    ])

    modified_embedding = x_font + embedding_extra_modification_model(
        tf.keras.layers.Concatenate()([x_extra, x_font])
    )

    embedding_size_modification_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(300, input_shape=(modified_embedding.shape[1] + x_size.shape[1],),
                              activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(modified_embedding.shape[1], activation='tanh'),
    ])

    modified_embedding = modified_embedding + embedding_size_modification_model(
        tf.keras.layers.Concatenate()([x_size, modified_embedding])
    )
    
    x = tf.keras.layers.Concatenate()([x_pos, modified_embedding])
    x = tf.keras.layers.Dense(1620, activation='relu')(x)
    x = tf.keras.layers.Dense(1082, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = x + tf.keras.layers.Dense(1082, activation='relu')(x)
    x = x + tf.keras.layers.Dense(1082, activation='relu')(x)
    x = x + tf.keras.layers.Dense(1082, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Concatenate()([x, modified_embedding])
    x = tf.keras.layers.Dense(1082, activation='relu')(x)
    x = x + tf.keras.layers.Dense(1082, activation='relu')(x)
    x = x + tf.keras.layers.Dense(1082, activation='relu')(x)
    x = x + tf.keras.layers.Dense(1082, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = x + tf.keras.layers.Dense(1082, activation='relu')(x)
    x = x + tf.keras.layers.Dense(1082, activation='relu')(x)
    x = tf.keras.layers.Dense(INTENSITY_CATEGORIES, activation='softmax')(x)
    y = tf.matmul(x, w)[:, 0]
    
    return tf.keras.models.Model(inputs=[x_size, x_pos, x_font, x_extra], outputs=[x, y, modified_embedding])


def make_model(n_glyphs, n_fonts):
    x_glyph = tf.keras.layers.Input((1,))
    x_size = tf.keras.layers.Input((2 * len(POSITIONAL_ENCODINGS) + 1,))
    x_pos = tf.keras.layers.Input((2 * (2 * len(POSITIONAL_ENCODINGS) + 1),))
    x_font = tf.keras.layers.Input((1,))
    x_extra = tf.keras.layers.Input((1,))
    inner_model = make_inner_model()

    font_embedding_layer = tf.keras.layers.Embedding(n_fonts, FONT_EMBEDDING_DIM, input_length=1)
    font_embedding = tf.keras.layers.Flatten()(font_embedding_layer(x_font))

    x, y, modified_embedding = inner_model([x_size, x_pos, font_embedding, x_extra])
    
    return (tf.keras.models.Model(inputs=[x_glyph, x_size, x_pos, x_font, x_extra], outputs=[x, y, modified_embedding]),
            inner_model,
            tf.keras.models.Sequential([font_embedding_layer]))


def make_discriminator_classifier():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(4, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(FONT_EMBEDDING_DIM + 1)
    ])

    model.build(input_shape=(None, None, None, 1))

    return model


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


def single_batch(model, discriminator, model_input,
                 actual_size, generator_optimizer, discriminator_optimizer,
                 batch_size, curr_Y_cat=None):
    with tf.GradientTape(persistent=True) as tape:
        pred_cat, pred, embedding = model(model_input, training=True)
                                    
        discriminator_pred = discriminator(tf.reshape(pred, (-1, actual_size, actual_size, 1)))

        embedding_pred = discriminator_pred[:, :-1]
        discriminator_pred = tf.sigmoid(discriminator_pred[:, -1])

        bce_loss_value = 0
        loss_denominator = EMBEDDING_LOSS_WEIGHT + DISCRIMINATOR_LOSS_WEIGHT
        discriminator_class = 0
        if curr_Y_cat is not None:
            bce_loss_value = (
                focal_loss_cce(curr_Y_cat, pred_cat)
                    + REG_LAMBDA * tf.reduce_mean(tf.reduce_sum(embedding ** 2, axis=1))
            )
            loss_denominator = 1

        embedding_idx = np.arange(0, embedding.shape[0], embedding.shape[0] // batch_size)
        embedding_loss = tf.reduce_mean(tf.norm(embedding_pred
                                        - tf.gather(embedding, embedding_idx), axis=1))

        discriminator_loss = bce(discriminator_class, discriminator_pred)

        loss_value = (
            bce_loss_value 
            + EMBEDDING_LOSS_WEIGHT * embedding_loss 
            # Negative for gradient reversal (want to fool discriminator here)
            - DISCRIMINATOR_LOSS_WEIGHT * discriminator_loss
        ) / loss_denominator

        discriminator_loss_value = (
            EMBEDDING_LOSS_WEIGHT * embedding_loss
            + DISCRIMINATOR_LOSS_WEIGHT * discriminator_loss
        )

    generator_grads = tape.gradient(loss_value, model.trainable_weights)
    generator_optimizer.apply_gradients(zip(generator_grads, model.trainable_weights))

    discriminator_grads = tape.gradient(discriminator_loss_value, discriminator.trainable_weights)
    discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator.trainable_weights))

    del tape

    return float(loss_value), float(discriminator_loss_value), float(bce_loss_value)
    

def run_experiment(subdirs, glyph_nums, glyph_metadata, bitmap_size, n_sizes,
                   base_names, modifier_names, RES_NAME,
                   model=None, n_epochs=1):
    if model is None:
        model, inner_model, font_embeddings = make_model(glyph_metadata.shape[0], len(subdirs))

    discriminator = make_discriminator_classifier()

    all_combinations = [
        (os.path.join(subdir, f'{glyph_num}.png'),
         glyph_num,
         subdir_idx, font_idx, e)
        for font_idx, (font, font_subdirs) in enumerate(subdirs.items())
        for e, e_subdirs in font_subdirs.items()
        for subdir_idx, subdir in enumerate(e_subdirs)
        for glyph_num in glyph_nums
    ]

    learning_rates = make_piecewise_lr_schedule(n_epochs)
    generator_optimizer_real = tf.keras.optimizers.Adam(learning_rate=learning_rates[0])
    generator_optimizer_fake = tf.keras.optimizers.Adam(learning_rate=learning_rates[0] / FAKE_OPTIMIZER_LR_RATIO)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rates[0])

    generator_loss_history = []
    discriminator_loss_history = []
    bce_loss_history = []
    batch_size = 4

    for epoch in range(1, n_epochs):
        if get_free_ram_percent() <= .075:
            print('Breaking due to low RAM!')
            break

        tf.keras.backend.set_value(generator_optimizer_real.learning_rate, learning_rates[epoch - 1])
        tf.keras.backend.set_value(generator_optimizer_fake.learning_rate, learning_rates[epoch - 1] / FAKE_OPTIMIZER_LR_RATIO)
        tf.keras.backend.set_value(discriminator_optimizer.learning_rate, learning_rates[epoch - 1])

        np.random.shuffle(all_combinations)
        all_combinations.sort(key=lambda x: x[2])

        all_sizes = np.array([c[2] for c in all_combinations])
        batch_idx = np.split(np.arange(len(all_combinations)), np.where(np.diff(all_sizes) != 0)[0] + 1)
        batches = [batch for g in batch_idx for batch in np.array_split(g, int(np.ceil(g.shape[0] / batch_size))) ]
        
        batch_generator_losses = []
        batch_discriminator_losses = []
        batch_bce_losses = []

        np.random.shuffle(batches)
        
        batch_pb = tqdm(batches, desc=f'Epoch {epoch}/{n_epochs}',
                        position=0, leave=True, smoothing=0)
        for idx, curr_batch_indices in enumerate(batch_pb):
            (curr_X_glyph, curr_X_size, curr_X_pos, curr_X_font,
             curr_X_extra, curr_Y_cat, curr_Y, actual_size) = load_batch([
                all_combinations[i]
                for i in curr_batch_indices
            ], n_sizes, glyph_metadata, bitmap_size, base_names, modifier_names)

            curr_batch_size = curr_batch_indices.shape[0]
            loss_value, discriminator_loss_value, bce_loss_value = \
                single_batch(model, discriminator,
                             [curr_X_glyph, curr_X_size, curr_X_pos, curr_X_font, curr_X_extra],
                             actual_size, generator_optimizer_real, discriminator_optimizer,
                             curr_batch_size, curr_Y_cat)

            batch_generator_losses.append(float(loss_value))
            batch_discriminator_losses.append(float(discriminator_loss_value))
            batch_bce_losses.append(float(bce_loss_value))

            random_font_X = tf.repeat(
                tf.random.normal((curr_batch_size, FONT_EMBEDDING_DIM)),
                repeats=curr_X_font.shape[0] // curr_batch_size,
                axis=0
            )

            random_extra_X = tf.repeat(
                tf.random.uniform((curr_batch_size, 1), 0, 1),
                repeats=curr_X_extra.shape[0] // curr_batch_size,
                axis=0
            )

            inner_model_input = [curr_X_size, curr_X_pos, random_font_X, random_extra_X]

            loss_value, discriminator_loss_value, bce_loss_value = \
                single_batch(inner_model, discriminator,
                            inner_model_input,
                            actual_size, generator_optimizer_fake, discriminator_optimizer,
                            curr_batch_size)

            # batch_generator_losses.append(float(loss_value))
            batch_discriminator_losses.append(float(discriminator_loss_value))
            # batch_bce_losses.append(float(bce_loss_value))

            with tf.GradientTape() as tape:
                discriminator_pred = discriminator(tf.reshape(curr_Y, (-1, actual_size, actual_size, 1)))
                discriminator_pred = tf.sigmoid(discriminator_pred[:, -1])
                discriminator_loss_value = bce(1, discriminator_pred)

            discriminator_grads = tape.gradient(discriminator_loss_value, discriminator.trainable_weights)
            discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator.trainable_weights))

            batch_discriminator_losses.append(float(discriminator_loss_value))

            batch_pb.set_postfix({
                'generator loss': np.mean(batch_generator_losses),
                'discriminator loss': np.mean(batch_discriminator_losses),
                'bce_loss': np.mean(batch_bce_losses),
            })

        batch_pb.close()

        generator_loss_history.append(np.mean(batch_generator_losses))
        discriminator_loss_history.append(np.mean(batch_discriminator_losses))
        bce_loss_history.append(np.mean(batch_bce_losses))

        if any([
            epoch == 1 and bce_loss_history[-1] >= 1,
            epoch == 10 and bce_loss_history[-1] >= .3,
            epoch == 50 and bce_loss_history[-1] >= .18,
            epoch == 300 and bce_loss_history[-1] >= .1
        ]):
            print('Breaking due to slow convergence!')
            break
        
    return (bce_loss_history, generator_loss_history,
            discriminator_loss_history, model, inner_model, font_embeddings)


def positional_encodings(v):
    return np.concatenate([
        [np.sin(pe * np.pi * v), np.cos(pe * np.pi * v)]
        for pe in POSITIONAL_ENCODINGS
    ])


def load_batch(batch, n_sizes, glyph_metadata, bitmap_size, base_names, modifier_names):
    X_glyph = []
    X_size = []
    X_pos = []
    X_font = []
    X_extra = []
    Y = []
    actual_size = []
    for img_path, ordered_glyph_idx, size_idx, font_idx, extra in batch:
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
                X_font.append(np.array([font_idx]))
                X_extra.append(np.array([extra]))
                actual_size.append(height)
        
        Y += y.ravel().tolist()

    X_glyph = np.stack(X_glyph)
    X_size = np.stack(X_size)
    X_pos = np.stack(X_pos)
    X_font = np.stack(X_font)
    X_extra = np.stack(X_extra)
    Y = np.stack(Y)
    actual_size = np.stack(actual_size)

    categorical_Y = np.zeros((Y.shape[0], INTENSITY_CATEGORIES))
    categorical_Y[range(Y.shape[0]), (np.floor(Y / (1 / (INTENSITY_CATEGORIES - 1)))).astype(int)] = 1

    return X_glyph, X_size, X_pos, X_font, X_extra, categorical_Y, Y, actual_size[0]


def get_free_ram_percent():
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t -m').readlines()[-1].split()[1:]
    )

    return free_memory / total_memory


def main():
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
            3/7: '/data/ground_truth/times_new_roman/times_new_roman_regular',
            6/7: '/data/ground_truth/times_new_roman/times_new_roman_bold'
        },
        'roboto': {
            1/7: '/data/ground_truth/roboto/roboto-thin',
            2/7: '/data/ground_truth/roboto/roboto-light',
            3/7: '/data/ground_truth/roboto/roboto-regular',
            4/7: '/data/ground_truth/roboto/roboto-medium',
            6/7: '/data/ground_truth/roboto/roboto-bold',
        },
        'hind': {
            2/7: '/data/ground_truth/hind/hind-light',
            3/7: '/data/ground_truth/hind/hind-regular',
            4/7: '/data/ground_truth/hind/hind-medium',
            5/7: '/data/ground_truth/hind/hind-semibold',
            6/7: '/data/ground_truth/hind/hind-bold',
        },
        'dancing_script': {
            3/7: '/data/ground_truth/dancing_script/dancingscript-regular',
            4/7: '/data/ground_truth/dancing_script/dancingscript-medium',
            5/7: '/data/ground_truth/dancing_script/dancingscript-semibold',
            6/7: '/data/ground_truth/dancing_script/dancingscript-bold',
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

    glyph_nums = GLYPH_NUMS
    sample_dir = list(list(roots.values())[0].values())[0]
    glyph_metadata = pd.read_csv(os.path.join(*os.path.split(sample_dir)[:-1], 'glyphs.csv'))
    glyph_metadata['modifier_indices'] = glyph_metadata['modifier_indices'].apply(json.loads)

    with open('/data/glyph_names.json', 'r') as f:
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

    RES_NAME = 'cap_a'

    os.makedirs(f'/data/training/v11_real_small/{RES_NAME}/model_checkpoints', exist_ok=True)
    os.makedirs(f'/data/results/v11_real_small/{RES_NAME}', exist_ok=True)

    (bce_hist, gen_hist, disc_hist,
     model, inner_model, font_embeddings) = run_experiment(subdirs, glyph_nums, glyph_metadata,
                                                           bitmap_size, n_sizes,
                                                           base_names, modifier_names, RES_NAME,
                                                           n_epochs=5)

    with open(f'/data/training/v11_real_small/{RES_NAME}/bce_loss_hist.json', 'w') as f:
        json.dump(bce_hist, f)

    with open(f'/data/training/v11_real_small/{RES_NAME}/generator_loss_hist.json', 'w') as f:
        json.dump(gen_hist, f)

    with open(f'/data/training/v11_real_small/{RES_NAME}/discriminator_loss_hist.json', 'w') as f:
        json.dump(disc_hist, f)

    model.save(f'/data/training/v11_real_small/{RES_NAME}/model')
    inner_model.save(f'/data/training/v11_real_small/{RES_NAME}/inner_model')
    font_embeddings.save(f'/data/training/v11_real_small/{RES_NAME}/font_embeddings')


if __name__ == '__main__':
    main()
