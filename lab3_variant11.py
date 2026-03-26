import os

import numpy as np
from PIL import Image

input_dir = 'input_images'
output_dir = 'output_images'
allowed = {'.bmp', '.png', '.jpg', '.jpeg'}

weights = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.uint8)
rank_value = 12
weight_sum = 16


def load_binary(path):
    img = Image.open(path).convert('L')
    arr = np.array(img, dtype=np.uint8)
    out = np.zeros_like(arr, dtype=np.uint8)
    out[arr > 127] = 255
    return out


def apply_filter(binary):
    h, w = binary.shape
    black = (binary == 0).astype(np.uint8)
    padded = np.pad(black, 1, mode='edge')
    result_black = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            window = padded[y:y + 3, x:x + 3]
            s = int((window * weights).sum())
            if s >= rank_value:
                result_black[y, x] = 1

    result = np.full((h, w), 255, dtype=np.uint8)
    result[result_black == 1] = 0
    return result


def xor_diff(a, b):
    return np.bitwise_xor(a, b)


def save(arr, path):
    Image.fromarray(arr).save(path)


def save_strip(inp, filt, diff, path):
    inp_rgb = np.stack([inp, inp, inp], axis=2)
    filt_rgb = np.stack([filt, filt, filt], axis=2)
    diff_rgb = np.stack([diff, diff, diff], axis=2)
    strip = np.concatenate([inp_rgb, filt_rgb, diff_rgb], axis=1)
    Image.fromarray(strip).save(path)


def main():
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    files = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        ext = os.path.splitext(name)[1].lower()
        if os.path.isfile(path) and ext in allowed:
            files.append(path)

    for path in files:
        img = load_binary(path)
        filtered = apply_filter(img)
        diff = xor_diff(img, filtered)

        base = os.path.splitext(os.path.basename(path))[0]
        save(filtered, os.path.join(output_dir, base + '_filtered.png'))
        save(diff, os.path.join(output_dir, base + '_xor.png'))
        save_strip(img, filtered, diff, os.path.join(output_dir, base + '_result.png'))

    print('done')


if __name__ == '__main__':
    main()
