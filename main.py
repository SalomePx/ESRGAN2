import tensorflow as tf
from datetime import datetime
import numpy as np
import os.path
import sys
from tqdm import tqdm

from data import create_patches, create_mito_mask, create_noised_inputs, RawData, load_training_data, load_training_data_from_patch
from config import *
from sixy import Path


def main():
    load = None
    log = tqdm.write

    # Create folder where model and prediction will be saved
    date = datetime.today().strftime('%m-%d-%H_%M_%S')
    name_add = TRAIN_LOSS if load is None else load
    model_savedir = os.path.join(f'{DIR_SAVE}', f'{date}_{name_add}')

    # Create the folder where each training launched will be saved
    if not os.path.isdir(DIR_SAVE):
        os.makedirs(DIR_SAVE)
    if not os.path.isdir(model_savedir):
        os.makedirs(model_savedir)
    else:
        print("Error: the folder where to save model already exist: retry.")

    # Extract GT and create LR data
    if DATA_NOISING:
        from utils import extract_files
        # The folder from where GT data are extracted must end by `_GT`
        extract_files(source_path=DIR_DATA, gt_folder_name=f"{DIR_DATA}_GT")
        create_noised_inputs(data_path=DIR_DATA,  gaussian_blur=3.25, gaussian_sigma=4, poisson_noise=True, pattern=DATA_INP_PAT)

    # Get list of LR test files to be predicted and exit if not found
    file_list = list(Path(f'{DIR_DATA}/test/low').glob(DATA_INP_PAT))
    if len(file_list) == 0:
        log("No files to process in '%s' with pattern '%s'." % (f'{DIR_DATA}/test/low', DATA_INP_PAT))
        sys.exit(0)

    # Process data augmentation (data duplication, transformation composition)
    from data import set_transformations
    transforms = set_transformations() if DATA_AUG else (None, None)

    # Build binary mask of area of interest for patch creation
    if not Path(DIR_CREAT_PATCHES).exists():
        create_mito_mask(data_dir=f"{DIR_DATA}/train/GT", saving_dir=f"{DIR_CREAT_PATCHES}/train")
        create_mito_mask(data_dir=f"{DIR_DATA}/test/GT", saving_dir=f"{DIR_CREAT_PATCHES}/test")

    # Create data if not created (patches)
    if CREAT_PATCHES:
        # Generator on (low, GT, mask_cristae, mask_mitochondria, mask_creation_of_patches)
        raw_data = RawData.from_folder(
            base_path=f'{DIR_DATA}/train',
            source_dirs=['low'], target_dir='GT',
            axes=AXES, mask_dirs=[DIR_MASK_CRISTAE, DIR_MASK_MITO, DIR_CREAT_PATCHES]
        )

        # Create patches
        if len(AXES) == 3: (PATCH_SIZE, PATCH_SIZE) = (PATCH_SIZE, PATCH_SIZE) + (3,)
        X, Y, C, M, XY_axes = create_patches(
            raw_data=raw_data, patch_size=tuple(PATCH_SIZE),
            data_path=DIR_DATA, saving_dir=model_savedir,
            axes=AXES, transforms=transforms,
            proportion=1, save_file=f'my_training_data.npz'
        )

        new_data_patch_dir = f"{model_savedir}/{str(DIR_DATA.name)}_patches"
        train_set, validation_set, axes = load_training_data(f'{new_data_patch_dir}/my_training_data.npz', validation_split=0.2)

    # Load patches already created
    elif load is None:
        # Load from .npz file
        # train_set, validation_set, axes = load_training_data_from_patch(file_name_npz, validation_split=0.2)

        # Load from patches
        train_set, validation_set, axes = load_training_data_from_patch(validation_split=0.2)

    # Training model
    if load is None:
        # Create train and validation data
        (X, Y, C, M), (X_val, Y_val, C_val, M_val) = train_set, validation_set

        # Add (mask_cristae, mask_mitochondria) to the GT data
        #Y_and_mask, Y_and_mask_val = np.concatenate((Y, C), axis=-1), np.concatenate((Y_val, C_val), axis=-1)
        #Y_and_mask, Y_and_mask_val = np.concatenate((Y_and_mask, M), axis=-1), np.concatenate((Y_and_mask_val, M_val), axis=-1)

        # Prepare the training dataset.
        train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

        # Prepare the validation dataset.
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        val_dataset = val_dataset.batch(BATCH_SIZE)

        from train import train
        # Train
        train(train_dataset, val_dataset, model_savedir)
        # Evaluate metrics
        #train_gan(X, Y_and_mask, X_val, Y_and_mask_val)

main()