import argparse

"""
Defines the Hyperparameters as command line arguments
"""

'''
For Parsing Booleans
'''
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

parser = argparse.ArgumentParser(description='Parser')

parser.add_argument('--task', type=str, default=None, help='seg, depth, or segdepth')
parser.add_argument('--backbone', type=str, default='b5', help='Backbone of the model')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of Workers')
parser.add_argument('--num_epochs', type=int, default=400, help='Number of Epochs')
parser.add_argument('--learning_rate', type=float, default=6e-5, help='Learning Rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight Decay')
parser.add_argument('--precision', type=str, default='16-mixed', help='Precision')
parser.add_argument('--devices', type=int, nargs='+', default=[1], help='Devices')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint')
parser.add_argument('--augmentations', type=str, nargs='+', default=['rand_flip', 'rand_scale'], help='Augmentations')
parser.add_argument('--name', type=str, default='default', help='Log Directory Name')
parser.add_argument('--version', type=str, default=None, help='Log Version')
parser.add_argument('--loss_seg_weight', type=float, default=1.0, help='Loss Segmentation Weight')
parser.add_argument('--loss_depth_weight', type=float, default=1.0, help='Loss Depth Weight')

args = parser.parse_args()

# Dataset
NUM_CLASSES = 40
IGNORE_INDEX = 255
NUMBER_TRAIN_IMAGES = 795
NUMBER_VAL_IMAGES = 654

# Model
TASK = args.task
BACKBONE = args.backbone
LOSS_SEG_WEIGHT = args.loss_seg_weight
LOSS_DEPTH_WEIGHT = args.loss_depth_weight

# Training
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
NUM_EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
WEIGHT_DECAY = args.weight_decay
PRECISION = args.precision
DEVICES = args.devices
CHECKPOINT = args.checkpoint
AUGMENTATIONS = args.augmentations
NAME = args.name
VERSION = args.version