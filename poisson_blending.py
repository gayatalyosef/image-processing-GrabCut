import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse


def timing_val(func):
    import time
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print(f'{func.__name__} took {(t2 - t1)}')
        return res
    return wrapper

@timing_val
def laplacian_matrix(rows, columns):
    D = scipy.sparse.identity(columns, format='lil') * 4
    D.setdiag(-1, -1)
    D.setdiag(-1, 1)

    laplacian = scipy.sparse.block_diag([D] * rows).tolil()
    laplacian.setdiag(-1, columns)
    laplacian.setdiag(-1, -columns)

    return laplacian


def is_boundary(i, j, rows, columns, im_mask):
    out_pixels = []
    if i == 0 or im_mask[i-1, j] == 0:
        out_pixels.append(columns*(i-1)+j)
    if i+1 == rows or im_mask[i+1, j] == 0:
        out_pixels.append(columns*(i+1)+j)
    if j == 0 or im_mask[i, j-1] == 0:
        out_pixels.append((columns*i+j-1))
    if j+1 == columns or im_mask[i, j+1] == 0:
        out_pixels.append((columns*i+j+1))
    return out_pixels

@timing_val
def build_A(im_src, im_mask):
    rows, columns = im_src.shape[:2]
    A = laplacian_matrix(*im_src.shape[:2])
    laplacian = A.copy()
    old_im_mask = im_mask.copy()

    for pixel_number in range(rows*columns):
        i = pixel_number // columns
        j = pixel_number % columns
        out_pixels = is_boundary(i, j, rows, columns, old_im_mask)
        if old_im_mask[i][j] == 0:
            A.rows[pixel_number] = [pixel_number]
            A.data[pixel_number] = [1]
        elif out_pixels:
            for out_pixel in out_pixels:
                index = A.rows[pixel_number].index(out_pixel)
                del A.rows[pixel_number][index]
                del A.data[pixel_number][index]
            im_mask[i][j] = 0

    return A.tocsc(), laplacian

@timing_val
def compare_imges_size(im_src, im_tgt, is_mask=False):
    src_rows, src_columns = im_src.shape[:2]
    tgt_rows, tgt_columns = im_tgt.shape[:2]
    add_to_rows = tgt_rows - src_rows
    add_to_columns = tgt_columns - src_columns

    if add_to_rows < 0 or add_to_columns < 0:
        raise Exception("Source is bigger than target")

    if is_mask:
        resized_source = np.zeros(im_tgt.shape[:2])
    else:
        resized_source = np.ones(im_tgt.shape)

    for i in range(src_rows):
        for j in range(src_columns):
            resized_source[i+add_to_rows//2][j+add_to_columns//2] = im_src[i][j]

    return resized_source

@timing_val
def build_B_of_color(laplacian, im_src, im_tgt, im_mask, color_index):

    _im_src = im_src[:, :, color_index].flatten()
    _im_tgt = im_tgt[:, :, color_index].flatten()
    _im_mask = im_mask.flatten()

    B = laplacian.dot(_im_src)
    # B = _im_src
    B[_im_mask == 0] = _im_tgt[_im_mask == 0]
    return B

@timing_val
def calculate_X(A, B, row, columns):
    # x = B
    x = spsolve(A, B)
    x = x.reshape((row, columns))
    x[x > 255] = 255
    x[x < 0] = 0
    return x


def calculate_blended_img(im_src, im_tgt, im_mask):
    A, L = build_A(im_src, im_mask)
    blended_img = im_tgt.copy()

    Br = build_B_of_color(L, im_src, im_tgt, im_mask, 0)
    Bg = build_B_of_color(L, im_src, im_tgt, im_mask, 1)
    Bb = build_B_of_color(L, im_src, im_tgt, im_mask, 2)
    Xr = calculate_X(A, Br, *im_tgt.shape[:-1])
    Xg = calculate_X(A, Bg, *im_tgt.shape[:-1])
    Xb = calculate_X(A, Bb, *im_tgt.shape[:-1])
    blended_img[:, :, :] = np.array([Xr, Xg, Xb]).transpose((1, 2, 0))
    return blended_img


def poisson_blend(im_src, im_tgt, im_mask, center):
    im_src = compare_imges_size(im_src, im_tgt)
    im_mask = compare_imges_size(im_mask, im_tgt, True)
    im_blend = calculate_blended_img(im_src, im_tgt, im_mask)
    return im_blend


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/book.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/book.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()

if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
