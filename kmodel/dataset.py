import numpy as np
import shutil
import os
import os.path as osp


def copy_as_keras_split(img_dir, tgt_dir, categ=None, val_perc=0.1,
                        cp_fn=shutil.copyfile):
    """Copy and split images from a source to a target dir to use with Keras

    :param img_dir: path to a folder with structure
    ```
    img_dir/
        catA/
           x1.jpg
           x2.jpg
           ...
        catB/
           y1.jpg
           y2.jpg
           ...
    ```
    :param tgt_dir: path to a different folder, ideally empty, where the
    images will be copied to following the Keras expected structure.
    :param categ: name of a category to copy. If `None`, it recursively
    copies all categories in the `img_dir`
    :param val_perc: percentage of images from each folder to hold back
    for validation. Float between 0.0 and 1.0
    :param cp_fn: function to copy files from inside `img_dir` to `tgt_dir`.
    The function must accept two parameters, the source path and the target
    path. By default, we use `shutil.copyfile`, but another option could be
    to use `os.symlink` to avoid copying files.
    :returns: a tuple with the training and validation folder paths. The
    target dir will have the structure expected by Keras ImageDataGenerator:
    ```
    tgt_dir/
        train/
           categA/
              x1.jpg
              ...
           categB/
              y1.jpg
              ...
        valid/
           categA/
              x2.jpg
              ...
           categB/
              y2.jpg
              ...
    ```
    :rtype: tuple of strings

    """
    if categ is None:
        for cd in os.listdir(img_dir):
            if osp.isdir(osp.join(img_dir, cd)):
                copy_as_keras_split(img_dir, tgt_dir, categ=cd)
        return osp.join(tgt_dir, 'train'), osp.join(tgt_dir, 'valid')

    # copy images for a specific category
    categ_dir = osp.join(img_dir, categ)

    def is_img(f):
        return osp.isfile(osp.join(categ_dir, f)) and f.endswith('jpg')

    jpgs = [f for f in os.listdir(categ_dir) if is_img(f)]
    print('found', len(jpgs), 'in', categ_dir)

    tr_dir = osp.join(tgt_dir, 'train', categ)
    va_dir = osp.join(tgt_dir, 'valid', categ)
    if not osp.exists(tr_dir):
        os.makedirs(tr_dir)
    if not osp.exists(va_dir):
        os.makedirs(va_dir)

    rnd_jpgs = np.random.permutation(jpgs)
    split = int(len(rnd_jpgs) * val_perc)
    print('splitting at', split)
    print('test files', len(rnd_jpgs[split:]))
    print('validations', len(rnd_jpgs[:split]))

    def clean_folder(in_dir):
        rmd = 0
        for f in os.listdir(in_dir):
            # print('removing previous ')
            path = os.path.join(in_dir, f)
            if os.path.isfile(path):
                os.remove(path)
                rmd = rmd + 1
        print('removed ', rmd, 'files from folder', in_dir)

    clean_folder(va_dir)
    clean_folder(tr_dir)

    for f in rnd_jpgs[:split]:
        # print('type', f, type(f), len(f))
        cp_fn(
            osp.join(categ_dir, f),  # src
            osp.join(va_dir, f))  # dst

    for f in rnd_jpgs[split:]:
        cp_fn(
            osp.join(categ_dir, f),  # src
            osp.join(tr_dir, f))  # dst

    tr_sl = [f for f in os.listdir(tr_dir)]
    va_sl = [f for f in os.listdir(va_dir)]
    print('Copied', len(tr_sl), 'training files to ', tr_dir)
    print('Copied', len(va_sl), 'validation files to ', va_dir)
    return tr_dir, va_dir
