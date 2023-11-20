from torch.utils.data import Dataset, DataLoader

from utils import _valid_low_high_percentiles, axes_dict, move_channel_for_backend, normalize_mi_ma, axes_check_and_normalize, _raise
from scipy.ndimage import median_filter
from skimage.transform import resize
from sixy import Path
from six import string_types


from itertools import chain
from tifffile import imread
from tqdm import tqdm
import numpy as np
import random
import shutil
import math
import cv2
import sys
import os

class CustomDataset(Dataset):
    def __init__(self, low_res, high_res):
        self.low_res = low_res
        self.high_res = high_res

    def __len__(self):
        return len(self.low_res)

    def __getitem__(self, idx):
        low_res_sample = self.low_res[idx]
        high_res_sample = self.high_res[idx]
        return low_res_sample, high_res_sample


############################################
###### METHOD OF SELECTION OF PATCHES ######
############################################

def load_training_data(file, validation_split=0, axes=None, n_images=None, verbose=True):
    """ Load training data from file in ``.npz`` format.
    The data file is expected to have the keys:
        - ``X``    : Array of training input images.
        - ``Y``    : Array of corresponding target images.
        - ``C``    : Array of corresponding cristae binary mask.
        - ``M``    : Array of corresponding mitochondria binary mask.
        - ``A``    : Array of corresponding area of interest binary mask.
        - ``axes`` : Axes of the training images.

    > Usable with DeepCristae version.

    Parameters
    ----------
    file : str
        File name
    validation_split : float
        Fraction of images to use as validation set during training.
    axes: str, optional
        Must be provided in case the loaded data does not contain ``axes`` information.
    n_images : int, optional
        Can be used to limit the number of images loaded from data.
    verbose : bool, optional
        Can be used to display information about the loaded images.

    Returns
    -------
    tuple(tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`),
    tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`), str )
        Returns two tuples (`X_train`, `Y_train`,`C_train`, `M_train`), (`X_val`, `Y_val`, `C_val`, `YM_val`) of
        training and validation sets and the axes of the input images.
        The tuple of validation data will be ``None`` if ``validation_split = 0``.
    """

    f = np.load(file, allow_pickle=True)
    X, Y, C, M = f['X'], f['Y'], f['C'], f['M']
    if axes is None:
        axes = f['axes']

    axes = axes_check_and_normalize(axes)
    assert X.ndim == Y.ndim == len(axes)
    assert 'C' in axes
    if n_images is None:
        n_images = X.shape[0]
    assert X.shape[0] == Y.shape[0] == C.shape[0] == M.shape[0]
    assert 0 < n_images <= X.shape[0]
    assert 0 <= validation_split < 1

    X, Y, C, M = X[:n_images], Y[:n_images], C[:n_images], M[:n_images]
    channel = axes_dict(axes)['C']

    if validation_split > 0:
        n_val = int(round(n_images * validation_split))
        n_train = n_images - n_val
        assert 0 < n_val and 0 < n_train
        X_t, Y_t, C_t, M_t = X[-n_val:], Y[-n_val:], C[-n_val:], M[-n_val:]
        X, Y, C, A = X[:n_train], Y[:n_train], C[:n_train], M[:n_train]
        assert X.shape[0] == n_train and X_t.shape[0] == n_val
        X_t = move_channel_for_backend(X_t, channel=channel)
        Y_t = move_channel_for_backend(Y_t, channel=channel)
        C_t = move_channel_for_backend(C_t, channel=channel)
        M_t = move_channel_for_backend(M_t, channel=channel)

    X = move_channel_for_backend(X, channel=channel)
    Y = move_channel_for_backend(Y, channel=channel)
    C = move_channel_for_backend(C, channel=channel)
    M = move_channel_for_backend(A, channel=channel)

    axes = axes.replace('C', '')  # remove channel
    if backend_channels_last():
        axes = axes + 'C'
    else:
        axes = axes[:1] + 'C' + axes[1:]

    data_val = (X_t, Y_t, C_t, M_t) if validation_split > 0 else None

    if verbose:
        ax = axes_dict(axes)
        n_train, n_val = len(X), len(X_t) if validation_split > 0 else 0
        image_size = tuple(X.shape[ax[a]] for a in axes if a in 'TZYX')
        n_dim = len(image_size)
        n_channel_in, n_channel_out = X.shape[ax['C']], Y.shape[ax['C']]

        print('Number of training images:\t', n_train)
        print('Number of validation images:\t', n_val)
        print('Image size (%dD):\t\t' % n_dim, image_size)
        print('Axes:\t\t\t\t', axes)
        print('Channels In/Out:\t\t', n_channel_in, '/', n_channel_out)
        print('=' * 66)

    return (X, Y, C, M), data_val, axes


def no_background_patches(threshold=11, get_zscore=False):

    def residual(y):
        down = y[1:, :-1]
        right = y[:-1, 1:]
        res = (2 * y[:-1, :-1] - down - right) / math.sqrt(6)
        return res

    def anscombe_tfm(y):
        y = np.where(y < 0, 0, y)
        tfm = 2 * np.sqrt(3 / 8 + y)
        return tfm

    def mad(y):
        y = np.abs(y)
        m = np.median(y)
        sigma = 1.4826 * m
        return sigma

    def std_approx(y, poisson=True):
        if poisson:
            y = anscombe_tfm(y)
        res = residual(y)
        std = mad(res)
        return std

    def _filter(y, dtype=np.float32):
        if dtype is not None:
            y = y.astype(dtype)

        # Make max filter patch_size smaller to avoid only few non-bg pixel close to image border
        zscore = (y - np.median(y)) / std_approx(y)
        zscore = np.where(zscore < 0, 0, zscore)

        if get_zscore:
            return zscore

        mask_filter = zscore > threshold
        return mask_filter

    return _filter


######################
##### TRANSFORMS #####
######################


from collections import namedtuple



class Transform(namedtuple('Transform', ('name', 'generator', 'size'))):

    @staticmethod
    def identity():
        def _gen(inputs):
            for d in inputs:
                yield d

        return Transform('Identity', _gen, 1)


def identity():

    def _generator(inputs):

        for x, y, c, m, a in inputs:
            yield x, y, c, m, a

    return Transform('Identity', _generator, 1)


def flip_vertical():
    def _generator(inputs):

        for x, y, c, m, a in inputs:
            x = np.fliplr(x)
            y = np.fliplr(y)
            c = np.fliplr(c)
            m = np.fliplr(m)
            a = np.fliplr(a)
            yield x, y, c, m, a

    return Transform('Flipped images vertically', _generator, 1)


def flip_horizontal():
    def _generator(inputs):

        for x, y, c, m, a in inputs:
            x = np.flipud(x)
            y = np.flipud(y)
            c = np.flipud(c)
            m = np.flipud(m)
            a = np.flipud(a)
            yield x, y, c, m, a

    return Transform('Flipped images horizontally', _generator, 1)


def flip(degree=1):
    def _generator(inputs):

        for x, y, c, m, a in inputs:
            x = np.rot90(x, k=degree)
            y = np.rot90(y, k=degree)
            c = np.rot90(c, k=degree)
            m = np.rot90(m, k=degree)
            a = np.rot90(a, k=degree)
            yield x, y, c, m, a

    return Transform(f'Flipped images of {90 * degree} degrees', _generator, 1)


def shrink():

    def _generator(inputs):

        for x, y, c, m, a in inputs:
            rdm_shrinking = round2(random.uniform(0.5, 0.8))
            shrink_on_x_axis = random.randint(0, 2)
            if shrink_on_x_axis:
                new_size = (int(np.ceil(x.shape[0] * rdm_shrinking)), x.shape[1])
            else:
                new_size = (x.shape[0], int(np.ceil(x.shape[1] * rdm_shrinking)))
            # Round up to even numbers
            new_size = tuple([n + (n % 2) for n in new_size])

            resize_kwargs = {'mode': 'edge', 'anti_aliasing': False, 'anti_aliasing_sigma': None, 'preserve_range': True, 'order': 0}
            x = resize(x, new_size, **resize_kwargs)
            y = resize(y, new_size, **resize_kwargs)
            c = resize(c, new_size, **resize_kwargs)
            m = resize(m, new_size, **resize_kwargs)
            a = resize(a, new_size, **resize_kwargs)

            yield x, y, c, m, a

    return Transform('Shrink images randomly on x-and-y-axis', _generator, 1)


def set_transformations(func_add=[flip, flip, flip], func_compose=[flip_vertical, flip_horizontal, shrink],
                        args_add=[1, 2, 3], args_compose=[(), (), ()]):

    transforms_add = [func(arg) for func, arg in zip(func_add, args_add)]
    transforms_compose = [func(*arg) for func, arg in zip(func_compose, args_compose)]

    transforms = (transforms_add, transforms_compose)
    return transforms


#########################
##### NORMALIZATION #####
#########################

def sample_percentiles(pmin=(1, 3), pmax=(99.5, 99.9)):
    _valid_low_high_percentiles(pmin) or _raise(ValueError(pmin))
    _valid_low_high_percentiles(pmax) or _raise(ValueError(pmax))
    pmin[1] < pmax[0] or _raise(ValueError())
    return lambda: (np.random.uniform(*pmin), np.random.uniform(*pmax))


def norm_percentiles(percentiles=sample_percentiles(), relu_last=False):
    if callable(percentiles):
        _tmp = percentiles()
        _valid_low_high_percentiles(_tmp) or _raise(ValueError(_tmp))
        get_percentiles = percentiles
    else:
        _valid_low_high_percentiles(percentiles) or _raise(ValueError(percentiles))
        get_percentiles = lambda: percentiles

    def _normalize(patches_x, patches_y, x, y, channel):
        pmins, pmaxs = zip(*(get_percentiles() for _ in patches_x))
        percentile_axes = None if channel is None else tuple((d for d in range(x.ndim) if d != channel))
        _perc = lambda a, p: np.percentile(a, p, axis=percentile_axes, keepdims=True)
        patches_x_norm = normalize_mi_ma(patches_x, _perc(x, pmins), _perc(x, pmaxs))
        if relu_last:
            pmins = np.zeros_like(pmins)
        patches_y_norm = normalize_mi_ma(patches_y, _perc(y, pmins), _perc(y, pmaxs))
        return patches_x_norm, patches_y_norm

    return _normalize


def shuffle_inplace(*arrays, seed=None):
    if len(set(len(a) for a in arrays)) != 1:
        raise ValueError("Input arrays must have the same length along the first axis.")

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed=seed)

    indices = rng.permutation(len(arrays[0]))
    for a in arrays:
        a[:] = a[indices]


#################################
###### CREATION OF PATCHES ######
#################################

def sample_patches_in_image():
    def func(datas, patch_size, **kwargs):
        n_samples = kwargs['n_samples']
        y, x, c, m, a = datas

        # Checks
        assert (each_patch_size > 0 and (type(each_patch_size) is int) for each_patch_size in patch_size)
        assert x.shape == y.shape and x.shape[0] >= patch_size[0] and x.shape[1] >= patch_size[1]
        assert n_samples > 0

        # Delineate the zone where the center of future selected patches is possible (with a margin of 1)
        border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in
                               zip(patch_size, y.shape)])  # This zone is the center of the image
        # Keep areas of interest only in the center of the image for the selection of patches
        valid_center_idx = np.where(a[border_slices])

        # There is no relevant information in the image
        n_valid = len(valid_center_idx[0])
        if n_valid == 0:
            raise ValueError("'patch_filter' didn't return any region to sample from")

        # Obtain the indices of the center points of the selected patch in the global image
        global_center_idx = choose_with_distance(valid_center_idx, n_samples)
        global_center_idx = [global_center_idx[i] + s.start for s, i in
                             zip(border_slices, np.arange(len(global_center_idx)))]

        # Construct the associated patches for (y,x)
        datas = y, x, c, m
        patches = [
            np.stack([data[tuple(slice(_r - (_p // 2), _r + _p - (_p // 2)) for _r, _p in zip(r, patch_size))] for r in
                      zip(*global_center_idx)]) for data in datas]
        # _r in the center of the patch
        # _p is the size of the patch
        # _r - (_p // 2) is the index of the beginning of the patch
        # _r + _p - (_p // 2) is the index of the end of the patch

        return patches

    return func


def create_patches(raw_data, patch_size, data_path, saving_dir, axes='XY', transforms=(None, None),
                   generate_patches=sample_patches_in_image(), proportion=1, save_file=None,
                   normalization=norm_percentiles(), shuffle=False, verbose=True):

    # Images and transforms
    transforms_add_init, transforms_compose_init = transforms
    if transforms_add_init is None and transforms_compose_init is None:
        transforms_add, transforms_compose = [], []
    else:
        transforms_add, transforms_compose = transforms_add_init, transforms_compose_init
    transforms_add, transforms_compose = list(transforms_add), list(transforms_compose)
    transforms_add.append(Transform.identity())

    if normalization is None:
        normalization = lambda patches_x, patches_y, x, y, channel: (patches_x, patches_y)

    # Create one generator of generators for all transforms
    nb_transforms_add = len(transforms_add)
    all_files_name = [f.split('.')[0] for f in sorted(os.listdir(f'{data_path}/train/GT'))]
    initial_nb_images = raw_data.size
    all_image_pairs = [raw_data.generator() for _ in range(nb_transforms_add)]
    tf_add = Transform(*zip(*transforms_add))

    # Augmentation (duplication of data with transformation)
    generators = [gen(image_pair) for gen, image_pair in zip(tf_add.generator, all_image_pairs)]
    image_pairs = chain(*generators)
    list_image_pairs = list(image_pairs)

    # Augmentation if selected (composition of transformation on data) else Identity
    if transforms_compose_init is not None:
        tf_compose = Transform(*zip(*transforms_compose))
        list_image_pairs = compose_randomly(*tf_compose.generator, list_data=list_image_pairs)

    # Summary of transforms
    if verbose:
        sep = '=' * 66
        print(f'{sep}\nInput data:\n{raw_data.description}\n{sep}\nTransformations:')
        [print(f'{t.size} x {t.name}') for t in transforms_add + transforms_compose]
        print(f'{sep}\nPatch size:\n{" x ".join(str(p) for p in patch_size)}\n{sep}')
    sys.stdout.flush()

    # Check memory for patches
    n_patches, n_images = memory_check_patches(list_image_pairs, patch_size, proportion)

    # Initialize sampling patches from each pair of transformed raw images
    X = np.empty((n_patches,) + tuple(patch_size), dtype=np.float32)
    Y, C, M = np.empty_like(X), np.empty_like(X), np.empty_like(X)

    # Create new directory of patches
    new_data_patch_dir = f"{data_path}_patches".split('/')[-1]
    new_data_patch_dir = f"{new_data_patch_dir}".split('\\')[-1]

    # --- Train directory
    name_train_dir = f"{new_data_patch_dir}/train"
    create_dir(f'{saving_dir}/{new_data_patch_dir}', ['train', 'mask_mito', 'mask_cristae'])
    create_dir([f'{saving_dir}/{name_train_dir}/GT', f'{saving_dir}/{name_train_dir}/low'])
    # --- Test directory: copy from initial data dir
    source_path = f"{data_path}/test"
    target_path = f"{saving_dir}/{new_data_patch_dir}/test"
    shutil.copytree(source_path, target_path)

    axes = axes_check_and_normalize(axes, len(patch_size))
    channel = axes_dict(axes)['C']

    # Create patches for each image
    occupied = 0
    for i, image_data in tqdm(enumerate(list_image_pairs), total=n_images, desc="Building data patches..."):
        x, y, c, m, a = check_tuple_format(image_data)
        idx, tfm = i % initial_nb_images, i // initial_nb_images
        image_name = all_files_name[idx].split('.')[0]
        patch_name = f'{image_name}_{str(tfm)}'

        # Checks
        axes == axes_check_and_normalize(axes) or _raise(ValueError('Not all images have the same axes.'))
        x.shape == y.shape or _raise(ValueError())
        (channel is None or (isinstance(channel, int) and 0 <= channel < x.ndim)) or _raise(ValueError())
        channel is None or patch_size[channel] == x.shape[channel] or _raise(
            ValueError('Extracted patches must contain all channels.'))

        # If image is big enough to create patches in
        if all(dimension >= size for dimension, size in zip(x.shape, patch_size)):

            # Compute number of patch on this
            count_patches_height, count_patches_width = (math.ceil(x.shape[i] / patch_size[i]) for i in range(2))
            num_patches_per_image = math.ceil(count_patches_height * count_patches_width * proportion)

            # Create patches
            _Y, _X, _C, _M = generate_patches((y, x, c, m, a), patch_size, n_samples=num_patches_per_image)

            # Add the selected patches if at least on patch is kept within the image
            if num_patches_per_image:
                s = slice(occupied, occupied + num_patches_per_image)
                X[s], Y[s] = normalization(_X, _Y, x, y, channel)
                C[s], M[s] = _C, _M

                # Save patches
                for k in range(len(_Y)):
                    idx = occupied + k
                    save_name = f'{patch_name}_{str(k)}'
                    pairs = (Y[idx], X[idx], C[idx], M[idx])
                    save_patch(pairs, save_name, saving_dir, new_data_patch_dir)

                occupied += num_patches_per_image

            else:
                print(f"No patches created for image: {image_name}")
        else:
            print(f"The image: {image_name} is too small to create patches of size: {patch_size}.")

    X, Y, C, M = X[:occupied], Y[:occupied], C[:occupied], M[:occupied]

    # Post process
    if shuffle:
        shuffle_inplace(X, Y, C, M)
    X, Y, C, M = (np.expand_dims(arr, 1) for arr in (X, Y, C, M)) if channel is None else (
        np.moveaxis(arr, 1 + channel, 1) for arr in (X, Y, C, M))
    axes = 'SC' + axes.replace('C', '')

    # Save patches into a `.npz` file working as a dico
    if save_file is not None:
        print('Saving data to %s.' % str(Path(save_file)))
        save_training_data(f"{saving_dir}/{new_data_patch_dir}/{save_file}", X, Y, C, M, axes)
    if verbose:
        print(f"There are {len(X)} final images for training and validation.")
        print('=' * 66)

    return X, Y, C, M, axes


###############################
###### CREATION OF MASKS ######
###############################

def create_mito_mask(data_dir, saving_dir, one_file=False, tshd_init=30, tshd_dec=5, perc_min=0.1):

    if not one_file:
        # Creation of the directory of masks
        create_dir(saving_dir, exist_ok=True)
        files_dir = os.listdir(data_dir)
        files = [f for f in files_dir if not os.path.isdir(os.path.join(data_dir, f))]

        # Accessing data from where to create masks
        path_data_masks = [(f"{data_dir}/{f}", f"{saving_dir}/{f.split('.')[0]}.png") for f in files]

    else:
        create_dir(saving_dir, exist_ok=True)
        file_name = data_dir.split('/')[-1]
        path_data_masks = [(data_dir, f"{saving_dir}/{file_name}")]

    # Checks
    n_images = len(path_data_masks)
    len(path_data_masks) > 0 or _raise(FileNotFoundError("Didn't find any images."))
    consume(Path(s).exists() or _raise(FileNotFoundError(Path(s))) for s, d in path_data_masks)

    # Checks
    n_images = len(path_data_masks)
    len(path_data_masks) > 0 or _raise(FileNotFoundError("Didn't find any images."))
    consume(Path(s).exists() or _raise(FileNotFoundError(Path(s))) for s, d in path_data_masks)

    # Create mask for each data
    for src_path, dst_path in tqdm(path_data_masks, total=n_images, desc="Creating masks of mitochondria..."):
        x = imread(src_path)
        tshd_tmp, nb_pixels_of_interest = tshd_init, 0
        occupation_min = perc_min * np.prod(x.shape)

        while nb_pixels_of_interest < occupation_min:
            boolean_mask = no_background_patches(tshd_tmp)(x)
            mask_no_filter = np.where(boolean_mask, 255, 0)
            mask = median_filter(mask_no_filter, size=(5, 5))
            nb_pixels_of_interest = np.sum(np.where(mask > 0, 1, 0))

            # Decrease the threshold requirement if not enough pixels of interest
            tshd_tmp -= tshd_dec

        cv2.imwrite(dst_path, mask)

def create_noised_inputs(data_path, source_dirs=['GT'], target_dir='low', gaussian_blur=3.25, gaussian_sigma=4,
                         poisson_noise=True, pattern='*.STED.ome.tif*'):

    p_train = Path(os.path.join(data_path, 'train'))
    p_test = Path(os.path.join(data_path, 'test'))

    images_path_train = sorted(chain(*((p_train / source_dir).glob(pattern) for source_dir in source_dirs)))
    images_path_test = sorted(chain(*((p_test / source_dir).glob(pattern) for source_dir in source_dirs)))
    images_path = [(f, p_train / target_dir / f.name) for f in images_path_train] + [(f, p_test / target_dir / f.name)
                                                                                     for f in images_path_test]

    if not images_path:
        raise FileNotFoundError("Didn't find any images to perturb.")

    for original_path, perturbed_path in tqdm(images_path, total=len(images_path), desc="Creating perturbated images... "):
        img = imread(original_path)
        img_noised = apply_noise(img, gaussian_blur, gaussian_sigma, poisson_noise)
        cv2.imwrite(str(perturbed_path), img_noised)


class RawData(namedtuple('RawData', ('generator', 'size', 'description'))):

    @staticmethod
    def from_folder(base_path, source_dirs, target_dir, mask_dirs, axes='YX', pattern='*.tif*'):

        p = Path(base_path)
        pairs = [(f, p / target_dir / f.name) for f in
                 sorted(chain(*((p / source_dir).glob(pattern) for source_dir in source_dirs)))]
        len(pairs) > 0 or _raise(FileNotFoundError("Didn't find any images."))
        consume(t.exists() or _raise(FileNotFoundError(t)) for s, t in pairs)
        axes = axes_check_and_normalize(axes)
        n_images = len(pairs)
        description = "{p}: target='{o}', sources={s}, pattern='{pt}'".format(p=base_path, s=list(source_dirs),
                                                                              o=target_dir, pt=pattern)

        def _gen():

            _pc, _pm, pa = mask_dirs
            pm = pa if _pm == Path('None') else _pm
            pc = pa if _pc == Path('None') else _pc

            pc = Path(f'{pc}/train')
            pm = Path(f'{pm}/train')
            pa = Path(f'{pa}/train')

            file_cristae = [pc / file for file in sorted(os.listdir(pc))]
            file_area = [pa / f for f in sorted(os.listdir(pa))]
            file_mito = [pm / f for f in sorted(os.listdir(pm))]

            file_mito != [] or _raise(ValueError("The path of your mitochondria mask is incorrect or is empty. If you "
                                                 "do not have one, set the parser argument --dirs-mask-mito to `None` (str)."))
            file_cristae != [] or _raise(ValueError("The path of your cristae mask is incorrect or is empty. If you do "
                                                    "not have one, set the parser argument --dirs-mask-cristae to `None` (str)."))

            for i, (low, gt) in enumerate(pairs):
                x, y = imread(str(low)), imread(str(gt))

                len(axes) >= x.ndim or _raise(ValueError())
                m = np.array(Image.open(str(file_mito[i])))
                a = np.array(Image.open(str(file_area[i])))
                if _pc is None:
                    c = np.ones(a.shape)
                else:
                    c = np.array(Image.open(str(file_cristae[i])))

                yield x, y, c, m, a

        return RawData(_gen, n_images, description)

    @staticmethod
    def from_arrays(X, Y, axes='YX'):
        """ Get pairs of corresponding images from numpy arrays."""

        def _gen():
            for x, y in zip(X, Y):
                len(axes) >= x.ndim or _raise(ValueError())
                yield x, y, axes[-x.ndim:], None

        return RawData(_gen, len(X), "numpy array")

    @staticmethod
    def from_mask(X, base_path):
        """ Get masks of images from numpy arrays."""

        p = Path(base_path)
        pairs = [(f, p / target_dir / f.name) for f in
                 sorted(chain(*((p / source_dir).glob(pattern) for source_dir in source_dirs)))]
        len(pairs) > 0 or _raise(FileNotFoundError("Didn't find any images."))
        consume(t.exists() or _raise(FileNotFoundError(t)) for s, t in pairs)

        def _gen():
            for x in X:
                yield x

        return RawData(_gen, len(X), "numpy array")


def save_training_data(file, X, Y, C=None, M=None, axes='XY'):

    isinstance(file, (Path, string_types)) or _raise(ValueError())
    file = Path(file).with_suffix('.npz')
    file.parent.mkdir(parents=True, exist_ok=True)

    axes = axes_check_and_normalize(axes)
    print(len(axes))
    print(X.ndim)
    len(axes) == X.ndim or _raise(ValueError())
    np.savez(str(file), X=X, Y=Y, C=C, M=M, axes=axes)



def load_training_data_from_patch(validation_split=0.2, axes='SCXY', patch_size=(128, 128), verbose=False):
    """ Load training data from patch images.
    The data is expected to have datas:
        - ``X``    : Array of training input images.
        - ``Y``    : Array of corresponding target images.
        - ``C``    : Array of corresponding cristae masks.
        - ``M``    : Array of corresponding mitochondria masks.

    > Made for DeepCristae version.

    Parameters
    ----------
    args: dict
        Command line parser arguments dictionary.
    validation_split : float
        Fraction of images to use as validation set during training.
    axes: str, optional
        Must be provided in case the loaded data does not contain ``axes`` information.
    patch_size : tuple (xaxis, yaxis)
        Size of the patches
    verbose : bool, optional
        Can be used to display information about the loaded images.

    Return
    ------
    tuple( tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`), tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`), str )
        Returns two tuples (`X_train`, `Y_train`), (`X_val`, `Y_val`) of training and validation sets
        and the axes of the input images.
        The tuple of validation data will be ``None`` if ``validation_split = 0``.
    """
    from PIL import Image
    import imageio
    from utils import axes_check_and_normalize, backend_channels_last
    from config import DIR_DATA

    images = os.listdir(f'{DIR_DATA}/train/GT')
    n_images = len(images)
    X = np.empty((n_images,) + tuple(patch_size), dtype=np.float32)
    #Y = np.empty((n_images, patch_size[0]*4, patch_size[1]*4))
    C, M = np.zeros_like(X), np.zeros_like(X)
    Y = np.empty((n_images,) + tuple(patch_size), dtype=np.float32)


    for i, im_path in tqdm(enumerate(images), total=n_images, desc="Loading images..."):
        name_patch = im_path.split('.tif')[0]
        f_read = Image.open if X.ndim == 4 else imread
        y = f_read(f'{DIR_DATA}/train/GT/{name_patch}.tif')
        x = f_read(f'{DIR_DATA}/train/low/{name_patch}.tif')
        c = imageio.imread(f'{DIR_DATA}/mask_cristae/{name_patch}.png')
        a = imageio.imread(f'{DIR_DATA}/mask_mito/{name_patch}.png')

        import imutils
        #y = cv2.resize(y, (512, 512))

        X[i], Y[i], C[i], M[i] = x, y, c, a

    channel = 1
    axes = axes_check_and_normalize(axes)
    X, Y, C, M = np.expand_dims(X, 1), np.expand_dims(Y, 1), np.expand_dims(C, 1), np.expand_dims(M, 1)

    assert 'C' in axes
    if n_images is None:
        n_images = X.shape[0]
    assert 0 < n_images <= X.shape[0]
    assert 0 <= validation_split < 1

    X, Y, C, M = X[:n_images], Y[:n_images], C[:n_images], M[:n_images]

    if validation_split > 0:
        n_val = int(round(n_images * validation_split))
        n_train = n_images - n_val
        assert 0 < n_val and 0 < n_train
        X_t, Y_t, C_t, M_t = X[-n_val:], Y[-n_val:], C[-n_val:], M[-n_val:]
        X, Y, C, M = X[:n_train], Y[:n_train], C[:n_train], M[:n_train]
        assert X.shape[0] == n_train and X_t.shape[0] == n_val
        X_t = move_channel_for_backend(X_t, channel=channel)
        Y_t = move_channel_for_backend(Y_t, channel=channel)
        C_t = move_channel_for_backend(C_t, channel=channel)
        M_t = move_channel_for_backend(M_t, channel=channel)

    X = move_channel_for_backend(X, channel=channel)
    Y = move_channel_for_backend(Y, channel=channel)
    C = move_channel_for_backend(C, channel=channel)
    M = move_channel_for_backend(M, channel=channel)

    axes = axes.replace('C', '')  # remove channel
    if backend_channels_last():
        axes = axes + 'C'
    else:
        axes = axes[:1] + 'C' + axes[1:]

    data_val = (X_t, Y_t, C_t, M_t) if validation_split > 0 else None

    if verbose:
        ax = axes_dict(axes)
        n_train, n_val = len(X), len(X_t) if validation_split > 0 else 0
        image_size = tuple(X.shape[ax[a]] for a in axes if a in 'TZYX')
        n_dim = len(image_size)
        n_channel_in, n_channel_out = X.shape[ax['C']], Y.shape[ax['C']]

        print('Number of training images:\t', n_train)
        print('Number of validation images:\t', n_val)
        print('Image size (%dD):\t\t' % n_dim, image_size)
        print('Axes:\t\t\t\t', axes)
        print('Channels In/Out:\t\t', n_channel_in, '/', n_channel_out)
        print('=' * 66)

    #save_training_data('my_training_data', X, Y, C=C, M=M, axes='XY')
    return (X, Y, C, M), data_val, axes

