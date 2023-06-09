import concurrent.futures
import logging
import math
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from BiLSTM import BiLSTMModel
from bilstm_training import load_txt_file_to_dataframe, get_bert_embeddings

# Disable the logging level for the transformers library
logging.getLogger("transformers").setLevel(logging.ERROR)

INPUT_SIZE: int = 768
SEQUENCE_LENGTH: int = 128
HIDDEN_SIZE: int = 64
NUM_LAYERS: int = 2
OUTPUT_SIZE: int = 1
EPOCHS: int = 250
BATCH_SIZE: int = 10
LEARNING_RATE: float = 0.001
CHKPT_INTERVAL: int = int(math.ceil(EPOCHS / 10))

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. PyTorch is using GPU:", torch.cuda.get_device_name(device))
else:
    print("GPU is not available. PyTorch is using CPU.")


multinli_df: pd.DataFrame = load_txt_file_to_dataframe("match")  # all

# Two lists of sentences for training
testA: List[str] = [x for x in multinli_df["sentence1"]]
testB: List[str] = [x for x in multinli_df["sentence2"]]

# Make labels
testY: List[int] = [1 if x == "contradiction" else 0 for x in multinli_df["gold_label"]]

testX: list = []
# Using ThreadPoolExecutor for parallel execution
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    # Calculate the total number of iterations
    total_iterations: int = len(testA)

    # Wrap the parallelized execution with tqdm
    with tqdm(total=total_iterations) as pbar:
        # Map the function to each pair of sentences in parallel
        results = executor.map(get_bert_embeddings, testA, testB)

        # Collect the results
        for result in results:
            testX.append(result)

            # Update progress bar
            pbar.update(1)

for i in range(len(testA)):
    print(f"{testA[i]} | {testA[i]}")
    print(f"{testY[i]}\n")

# Create training and dev sets
test_x: torch.Tensor = torch.stack([testX[i] for i in range(len(testX))]).view(len(testX), 128, 768)  # reshape to 3d

model = BiLSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
model.load_state_dict(torch.load("Models/model3.pth", map_location=device).state_dict())
model.eval()

with torch.no_grad():
    output = model(test_x.to(device))
    max_pooling_output, _ = torch.max(output, dim=1)
    print(max_pooling_output)
    avg_pooling_output = torch.mean(output, dim=1)
    print(avg_pooling_output)
    probabilities = torch.softmax(output, dim=1)
    predicted_labels = torch.argmax(probabilities, dim=1)
    output_np = predicted_labels.cpu().numpy()

print(f"Actual:      {np.array(testY)}")
print(f"Predictions: {output_np}")

accuracy = np.mean(np.array(testY) == output_np)
score = accuracy.item()

print(f"Score: {score}")
