from configs.data import SOURCE_DATA_CONFIG,TARGET_DATA_CONFIG, EVAL_DATA_CONFIG, TARGET_SET, NUM_CLASSES
from ever.magic.transform.segm import Scale

MODEL = dict(
    encoder=dict(
        name='resnet50',
        weights='imagenet',
        in_channels=3,
    ),
    fpn=dict(
        in_channels_list=(256, 512, 1024, 2048),
        out_channels=256,
    ),
    decoder=dict(
        in_channels=256,
        out_channels=128,
        in_feat_output_strides=(4, 8, 16, 32),
        out_feat_output_stride=4,
    ),
    classes=NUM_CLASSES,
)


IGNORE_LABEL = -1
MOMENTUM = 0.9
NUM_CLASSES = NUM_CLASSES

SAVE_PRED_EVERY = 4000

SNAPSHOT_DIR = './lovecs_sfpn'

#Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 1e-2
NUM_STEPS = 15000
NUM_STEPS_STOP = 15000  # Use damping instead of early stopping
PREHEAT_STEPS = 0
POWER = 0.9
EVAL_EVERY=2000

THRESH = 0.7
DS_RATE = 16
KC_VALUE = 'conf'
KC_POLICY = 'cb'
MINE_PORT = 1e-3
RARE_CLS_NUM = 1
RM_PROB = True
WARMUP_STEP = 10000
GENERATE_PSEDO_EVERY=2000
TGT_PORTION_STEP = 0.
SOURCE_LOSS_WEIGHT = 1.0
PSEUDO_LOSS_WEIGHT = 0.5

TGT_PORTION = 0.1
TGT_PORTION_STEP = 0.
MAX_TGT_PORTION = 0.1


TARGET_SET = TARGET_SET
SOURCE_DATA_CONFIG=SOURCE_DATA_CONFIG
TARGET_DATA_CONFIG=TARGET_DATA_CONFIG
EVAL_DATA_CONFIG=EVAL_DATA_CONFIG
PSETTA = True
TTA_LIST = [Scale(scale_factor=2.0), Scale(scale_factor=2.5), Scale(scale_factor=3.0), Scale(scale_factor=3.5)]