import freetype
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import pandas as pd
import json
from list_supported_glyphs import get_supported_glyphs


METRIC_NAMES = ['height', 'horiAdvance', 'horiBearingX', 'horiBearingY', 'width']


with open('/data/glyph_names.json', 'r') as f:
    names = json.load(f)
    base_names = names['base']
    modifier_names = names['modifiers']


glyphs = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'
              'abcdefghijklmnopqrstuvwxyz'
              '0123456789!@#$%^&*()-=_+:;/\\[]{}",.<>\'|`~')


def draw_glyph(face, size, glyph, resolution=72):
    face.set_char_size(0, int(size * 64), hres=resolution, vres=resolution)
    face.load_char(glyph)
    bmp = np.array(face.glyph.bitmap.buffer).reshape((face.glyph.bitmap.rows, face.glyph.bitmap.width))
    metrics = {
        metric: getattr(face.glyph.metrics, metric) / 64
        for metric in METRIC_NAMES
    }

    return bmp, metrics


def prep_draw_all_glyphs(face, size, resolution=72):
    bmps = []
    metrics = []
    for glyph in glyphs:
        curr_bmp, curr_metrics = draw_glyph(face, size, glyph, resolution)
        bmps.append(curr_bmp)
        metrics.append(curr_metrics)

    return bmps, find_bbox(metrics), metrics


def find_bbox(all_metrics):
    x_min = x_max = y_min = y_max = 0
    for metric in all_metrics:
        curr_x_min = metric['horiBearingX']
        curr_x_max = curr_x_min + metric['width']
        curr_y_max = metric['horiBearingY']
        curr_y_min = curr_y_max - metric['height']
        x_min = min(x_min, curr_x_min)
        x_max = max(x_max, curr_x_max)
        y_min = min(y_min, curr_y_min)
        y_max = max(y_max, curr_y_max)

    return (x_min, x_max, y_min, y_max)



def test_fit_size_resolution(face, size, square_size, resolution):
    bmps, bbox, metrics = prep_draw_all_glyphs(face, size, resolution)
    (x_min, x_max, y_min, y_max) = bbox
    w = x_max - x_min
    h = y_max - y_min
    return w <= square_size and h <= square_size, bmps, bbox, metrics


def find_largest_size_resolution_that_fits(face, square_size):
    curr_size = 0
    curr_resolution = 72
    fits = True
    bmps = bbox = metrics = None
    next_bmps = next_bbox = next_metrics = None
    while fits:
        bmps = next_bmps
        bbox = next_bbox
        metrics = next_metrics
        curr_size += 1
        fits, next_bmps, next_bbox, next_metrics = test_fit_size_resolution(face, curr_size, square_size, curr_resolution)

    curr_size -= 1
    fits = True
    next_bmps = next_bbox = next_metrics = None
    while fits:
        if next_bmps is not None:
            bmps = next_bmps
            bbox = next_bbox
            metrics = next_metrics
        curr_resolution += 1
        fits, next_bmps, next_bbox, next_metrics = test_fit_size_resolution(face, curr_size, square_size, curr_resolution)

    curr_resolution -= 1

    return curr_size, curr_resolution, bmps, bbox, metrics


def find_largest_size_resolution_that_fits_multiple(faces, square_size):
    size = resolution = float('inf')
    for face in faces:
        curr_size, curr_resolution, _, _, _ = find_largest_size_resolution_that_fits(face, square_size)
        if curr_size < size:
            size = curr_size
            resolution = curr_resolution
        elif size == curr_size and curr_resolution < resolution:
            resolution = curr_resolution

    res = []
    for face in faces:
        _, bmps, bbox, metrics = test_fit_size_resolution(face, size, square_size, resolution)
        res.append((size, resolution, bmps, bbox, metrics))

    return res


def generate_ground_truth(faces, square_size, face_paths):
    res = find_largest_size_resolution_that_fits_multiple(faces, square_size)
    for curr_res, face_path in zip(res, face_paths):
        point_size, resolution, bmps, bbox, metrics = curr_res
        x_min, x_max, y_min, y_max = bbox
        width = int(x_max - x_min)
        height = int(y_max - y_min)
        dir_path = os.path.join(face_path, f'{square_size}_pointsize_{point_size}_resolution_{resolution}')
        os.makedirs(dir_path, exist_ok=True)
        for i, (bmp, metric, glyph) in enumerate(zip(bmps, metrics, glyphs)):
            img = np.ones((square_size, square_size))
            col_start = int(metric['horiBearingX'] - x_min)
            col_end = int(col_start + metric['width'])
            row_start = height - int(metric['horiBearingY'] - y_min)
            row_end = int(row_start + metric['height'])
            img[row_start:row_end, col_start:col_end] = bmp
            img = 255 - img
            img = Image.fromarray(img.astype(np.uint8))
            img.save(os.path.join(dir_path, f'{ord(glyph)}.png'))


# def generate_ground_truth(face, square_size, face_path):
#     point_size, resolution, bmps, bbox, metrics = find_largest_size_resolution_that_fits(face, square_size)
#     x_min, x_max, y_min, y_max = bbox
#     width = int(x_max - x_min)
#     height = int(y_max - y_min)
#     dir_path = os.path.join(face_path, f'{square_size}_pointsize_{point_size}_resolution_{resolution}')
#     os.makedirs(dir_path, exist_ok=True)
#     for i, (bmp, metric, glyph) in enumerate(zip(bmps, metrics, glyphs)):
#         # img = np.ones((height, width))
#         img = np.ones((square_size, square_size))
#         col_start = int(metric['horiBearingX'] - x_min)
#         col_end = int(col_start + metric['width'])
#         row_start = height - int(metric['horiBearingY'] - y_min)
#         row_end = int(row_start + metric['height'])
#         img[row_start:row_end, col_start:col_end] = bmp
#         img = 255 - img
#         img = Image.fromarray(img.astype(np.uint8))
#         img.save(os.path.join(dir_path, f'{ord(glyph)}.png'))


FONT_ROOTS = [
    '/data/fonts/roboto/',
    '/data/fonts/arsenica/',
    '/data/fonts/times_new_roman',
    '/data/fonts/hind',
    '/data/fonts/dancing_script',
    '/data/fonts/roboto_slab'
]

for font_root in FONT_ROOTS:
    face_paths = [os.path.join(font_root, font) for font in os.listdir(font_root)]
    faces = [freetype.Face(fpath) for fpath in face_paths]

    # face = freetype.Face(os.path.join('/Users/andersource/Library/Fonts/', font))
    font_name = os.path.split(font_root.strip('/'))[-1].replace('.ttf', '').lower().replace(' ', '_')
    font_glyphs, font_names = get_supported_glyphs(face_paths[0])
    font_glyphs, font_names = zip(*[
        (glyph, name)
        for glyph, name in zip(font_glyphs, font_names)
        if glyph in glyphs
    ])

    glyphs = font_glyphs
    names = font_names

    df = pd.DataFrame({'idx': map(ord, glyphs), 'name': names})
    df['base_name'] = df['name'].str.upper().str.split('WITH').apply(lambda x: x[0]).str.strip()
    df['base_name_idx'] = df['base_name'].apply(base_names.index)
    df['modifiers'] = (df['name'].str.upper()
                       .str.split('WITH')
                       .apply(lambda x: x[1] if len(x) > 1 else '')
                       .str.split('AND')
                       .apply(lambda x: list(map(str.strip, x)))
                       .apply(lambda x: x if x != [''] else []))

    df['modifier_indices'] = df['modifiers'].apply(lambda x: list(map(modifier_names.index, x))).apply(json.dumps)
    font_paths = [
        os.path.join('/data/ground_truth', font_name,
                     font.replace('.ttf', '').lower().replace(' ', '_'))
        for font in os.listdir(font_root)
        ]

    font_path = os.path.join('/data/ground_truth', font_name)
    os.makedirs(font_path, exist_ok=True)
    df.to_csv(os.path.join(font_path, 'glyphs.csv'), index=False)

    for square_size in tqdm(range(20, 64), desc=f'Generating for {font_name}'):
        generate_ground_truth(faces, square_size, font_paths)

