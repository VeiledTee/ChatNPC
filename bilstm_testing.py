import math

from typing import Tuple, Dict, List
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from transformers import BertModel, BertTokenizer
import logging
from torch.utils.data import DataLoader, TensorDataset
import os

from tqdm import tqdm
import concurrent.futures

from BiLSTM import BiLSTMModel
import pandas as pd
import math

from BiLSTM import BiLSTMModel
from bilstm_training import load_txt_file_to_dataframe, get_bert_embeddings

# from transformers import LlamaForCausalLM
#
# model_path = 'openlm-research/open_llama_3b'
# # model_path = 'openlm-research/open_llama_7b'
#
# tokenizer = LlamtestAokenizer.from_pretrained(model_path)
# model = LlamaForCausalLM.from_pretrained(
#     model_path, torch_dtype=torch.float16, device_map='auto',
# )
#
# prompt = 'Q: What is the largest animal?\nA:'
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#
# generation_output = model.generate(
#     input_ids=input_ids, max_new_tokens=32
# )
# print(tokenizer.decode(generation_output[0]))
#
# # Example of target with class indices
# loss = nn.CrossEntropyLoss()
# input = torch.tensor([[0.0166],
#         [0.0449],
#         [0.0274],
#         [0.0522]], requires_grad=False)
# target = torch.tensor([[0],
#         [1],
#         [1],
#         [0]], requires_grad=False)
#
# # Create a tensor of zeros with the same number of samples
# zeros_tensor = torch.zeros(input.shape[0], 1, requires_grad=False)
# # Concatenate the predictions tensor with the zeros tensor
# input = torch.cat((input, zeros_tensor), dim=1)
# target = target.squeeze()
# print(input)
# print(target)
# output = loss(input, target)
# print(output)
# output.backward()
# import torch
#
# # setting device on GPU if available, else CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)
# print()
#
# #Additional Info when using cuda
# if device.type == 'cuda':
#     print(torch.cuda.get_device_name(0))
#     print('Memory Usage:')
#     print('Allocated:', torch.cuda.memory_allocated(0), 'GB')
#     print('Cached:   ', torch.cuda.memory_reserved(0), 'GB')
#
#
# print(torch.cuda.is_available())
#
# # Check if GPU is available
# if torch.cuda.is_available():
#     device = torch.device("cuda")  # Create CUDA device object
#     print("GPU is available. PtestYorch is using GPU:", torch.cuda.get_device_name(device))
# else:
#     device = torch.device("cpu")
#     print("GPU is not available. PtestYorch is using CPU.")
#
# # Move tensors and models to the GPU
# tensor = torch.tensor([1, 2, 3])  # Create a tensor
# tensor = tensor.to(device)  # Move tensor to the device (GPU or CPU)

# Define hyperparameters
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
