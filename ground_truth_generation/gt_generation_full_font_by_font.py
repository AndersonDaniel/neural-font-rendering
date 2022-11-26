import freetype
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import pandas as pd
import json
from list_supported_glyphs import get_supported_glyphs
import re


METRIC_NAMES = ['height', 'horiAdvance', 'horiBearingX', 'horiBearingY', 'width']


with open('/home/ubuntu/data/glyph_names.json', 'r') as f:
    names = json.load(f)
    base_names = names['base']
    modifier_names = names['modifiers']


actual_glyphs = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
# glyphs = get_supported_glyphs('/Users/andersource/Library/Fonts/yrsa/yrsa-bold.ttf')
glyphs = None


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
    # return all([
    #     (np.array(draw_glyph(face, size, glyph, resolution).shape) <= square_size).all()
    #     for glyph in glyphs
    # ])

    bmps, bbox, metrics = prep_draw_all_glyphs(face, size, resolution)
    (x_min, x_max, y_min, y_max) = bbox
    w = x_max - x_min
    h = y_max - y_min
    return w <= square_size and h <= square_size, bmps, bbox, metrics


def find_largest_size_resolution_that_fits(face, square_size, ptsize, resolution):
    fits, bmps, bbox, metrics = test_fit_size_resolution(face, ptsize, square_size, resolution)

    return ptsize, resolution, bmps, bbox, metrics


def generate_ground_truth(face, square_size, face_path, by_font_dirpath, ptsize, resolution):
    point_size, resolution, bmps, bbox, metrics = find_largest_size_resolution_that_fits(face, square_size, ptsize, resolution)
    x_min, x_max, y_min, y_max = bbox
    width = int(x_max - x_min)
    height = int(y_max - y_min)
    dir_path = os.path.join(face_path, f'{square_size}_pointsize_{point_size}_resolution_{resolution}')
    os.makedirs(dir_path, exist_ok=True)
    for i, (bmp, metric, glyph) in enumerate(zip(bmps, metrics, glyphs)):
        by_img = Image.open(os.path.join(by_font_dirpath, f'{ord(glyph)}.png'))
        by_img = 1 - np.asarray(by_img) / 255
        
        img = np.zeros((square_size, square_size))
        col_start = int(metric['horiBearingX'] - x_min)
        col_end = int(col_start + metric['width'])
        row_start = height - int(metric['horiBearingY'] - y_min)
        row_end = int(row_start + metric['height'])
        img[row_start:row_end, col_start:col_end] = bmp
        img = (align_images(img / 255, by_img) * 255).astype(np.uint8)
        img = 255 - img
        img = Image.fromarray(img.astype(np.uint8))
        img.save(os.path.join(dir_path, f'{ord(glyph)}.png'))


def align_images(img, anchor):
    max_alignment = -1
    max_aligned = None
    for row_offset in range(-10, 11):
        for col_offset in range(-10, 11):
            curr_img = shift_image(img, row_offset, col_offset)
            curr_alignment = (curr_img * anchor).sum()
            if curr_alignment > max_alignment:
                max_alignment = curr_alignment
                max_aligned = curr_img

    return max_aligned


def shift_image(img, row_offset, col_offset):
    s = img.shape[0]
    new_img = img.copy()
    sample_row = 0
    if row_offset < 0:
        sample_row = -row_offset
        new_img = np.concatenate([img, np.zeros((-row_offset, s))], axis=0)
    elif row_offset > 0:
        new_img = np.concatenate([np.zeros((row_offset, s)), img], axis=0)

    h = s + np.abs(row_offset)

    sample_col = 0
    if col_offset < 0:
        sample_col = -col_offset
        new_img = np.concatenate([new_img, np.zeros((h, -col_offset))], axis=1)
    elif col_offset > 0:
        new_img = np.concatenate([np.zeros((h, col_offset)), new_img], axis=1)

    return new_img[sample_row:sample_row + s, sample_col:sample_col + s]


FONTS = [
    # '/home/ubuntu/data/fonts/times_new_roman.ttf',
    # '/home/ubuntu/data/fonts/tahoma.ttf',
    # '/home/ubuntu/data/fonts/arial.ttf'
    '/home/ubuntu/data/fonts/lucida_sans.ttf',
]

BY_FONT = '/home/ubuntu/data/fonts/lucida_bright.ttf'

by_font_name = BY_FONT.split('/')[-1].replace('.ttf', '').lower().replace(' ', '_')
by_font_path = os.path.join('/home/ubuntu/data/ground_truth', by_font_name)

by_font_dirnames = {int(dirname.split('_')[0]): dirname for dirname in os.listdir(by_font_path)
                    if os.path.isdir(os.path.join(by_font_path, dirname))}

by_font_sizes = [list(map(int, re.search(r'(\d+)_pointsize_(\d+)_resolution_(\d+)', dirname).groups()))
                 for dirname in list(by_font_dirnames.values())]
by_font_sizes = {
    a: (b, c)
    for a, b, c in by_font_sizes
}

for font in FONTS:
    # face = freetype.Face(os.path.join('/Users/andersource/Library/Fonts/', font))
    font_name = font.split('/')[-1].replace('.ttf', '').lower().replace(' ', '_')
    glyphs, names = get_supported_glyphs(font)

    new_glyphs = []
    new_names = []
    for glyph, name in zip(glyphs, names):
        if glyph in actual_glyphs:
            new_glyphs.append(glyph)
            new_names.append(name)

    glyphs = new_glyphs
    names = new_names
    
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
    font_path = os.path.join('/home/ubuntu/data/ground_truth', font_name)
    os.makedirs(font_path, exist_ok=True)
    df.to_csv(os.path.join(font_path, 'glyphs.csv'), index=False)

    face = freetype.Face(font)

    for square_size in tqdm(range(20, 64), desc=f'Generating for {font_name}'):
        generate_ground_truth(face, square_size, font_path,
                              os.path.join(by_font_path, by_font_dirnames[square_size]),
                              *by_font_sizes[square_size])

