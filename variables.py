import math
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


#  Specify dataset to use (train/match/mismatch)
DATASET = "train"

# Define hyperparameters
INPUT_SIZE: int = 768
SEQUENCE_LENGTH: int = 128
HIDDEN_SIZE: int = 64
NUM_LAYERS: int = 2
OUTPUT_SIZE: int = 1
EPOCHS: int = 10
LEARNING_RATE: float = 0.001
CHKPT_INTERVAL: int = int(math.ceil(EPOCHS / 10))
TESTSET = "mismatch"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
if DATASET == "train":
    BATCH_SIZE: int = 1024
else:
    BATCH_SIZE: int = 256
# Set pytorch device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)