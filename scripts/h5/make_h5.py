import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
import cv2
# from ..data.utils import modcrop

def modcrop(img: np.ndarray, scale: int) -> np.ndarray:
    if img.ndim == 3:
        img_size = np.array(img.shape[0:2])
        img_size = img_size - np.mod(img_size, scale)
        out_img = img[0:img_size[0], 0: img_size[1], :]
    elif img.ndim == 2:
        img_size = np.array(img.shape[0:2])
        img_size = img_size - np.mod(img_size, scale)
        out_img = img[0:img_size[0], 0: img_size[1]]
    else:
        raise ValueError
    return out_img

def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_patches = []
    hr_patches = []

    for image_path in sorted(glob.glob('{}/*.bmp'.format(args.images_dir))):
        hr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        hr_mod_crop = modcrop(hr, args.scale)
        lr = cv2.resize(hr_mod_crop, dsize=None, fx=1/2, fy=1/2, interpolation=cv2.INTER_CUBIC)
        hr_float = np.array(hr_mod_crop).astype(np.float32)
        lr_float = np.array(lr).astype(np.float32)

        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                hr_patches.append(hr_float[i*args.scale:i*args.scale + args.patch_size*args.scale, j*args.scale:j*args.scale + args.patch_size*args.scale])
                lr_patches.append(lr_float[i:i + args.patch_size, j:j + args.patch_size])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*.bmp'.format(args.images_dir)))):
        # hr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        hr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        hr_mod_crop = modcrop(hr, args.scale)
        lr = cv2.resize(hr_mod_crop, dsize=None, fx=1/2, fy=1/2, interpolation=cv2.INTER_CUBIC)
        hr_float = np.array(hr_mod_crop).astype(np.float32)
        lr_float = np.array(lr).astype(np.float32)

        lr_group.create_dataset(str(i), data=lr_float)
        hr_group.create_dataset(str(i), data=hr_float)

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--patch-size', type=int, default=33)
    parser.add_argument('--stride', type=int, default=14)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)