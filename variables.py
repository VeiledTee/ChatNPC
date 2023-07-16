import math
import torch

#  Specify dataset to use (train/match/mismatch)
DATASET = "match"

# Define hyperparameters
INPUT_SIZE: int = 768
SEQUENCE_LENGTH: int = 128
HIDDEN_SIZE: int = 64
NUM_LAYERS: int = 2
OUTPUT_SIZE: int = 1
EPOCHS: int = 250
LEARNING_RATE: float = 0.001
CHKPT_INTERVAL: int = int(math.ceil(EPOCHS / 10))
TESTSET = 'match'
if DATASET == "train":
    BATCH_SIZE: int = 1024
else:
    BATCH_SIZE: int = 256
# Set pytorch device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
