import torch

LOAD_MODEL = None
SAVE_MODEL = True
CHECKPOINT_GEN = "gen.pth"
CHECKPOINT_DISC = "disc.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10000
BATCH_SIZE = 16
LAMBDA_GP = 10
NUM_WORKERS = 4
HIGH_RES = 128
LOW_RES = HIGH_RES
IMG_CHANNELS = 1
TRAIN_LOSS = 'GAN'
AXES = 'YX'
PATCH_SIZE = 128

DIR_SAVE = 'fig'
DATA_NOISING = False
DIR_DATA = 'data/data_mito_patch_deconv_rl2'
DIR_MASK_MITO = 'data/mask_mito'
DIR_MASK_CRISTAE = 'data/mask_cristae'


DIR_CREAT_PATCHES = 'data/creation_of_patches'
CREAT_PATCHES = False

DIR_ARCHIVES = 'archives/'
DATA_INP_PAT = '*.tif*'

DATA_AUG = True
