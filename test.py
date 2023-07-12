import torch
from torch import nn
# from transformers import LlamaTokenizer, LlamaForCausalLM
#
# model_path = 'openlm-research/open_llama_3b'
# # model_path = 'openlm-research/open_llama_7b'
#
# tokenizer = LlamaTokenizer.from_pretrained(model_path)
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

import torch

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', torch.cuda.memory_allocated(0), 'GB')
    print('Cached:   ', torch.cuda.memory_reserved(0), 'GB')


print(torch.cuda.is_available())

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Create CUDA device object
    print("GPU is available. PyTorch is using GPU:", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("GPU is not available. PyTorch is using CPU.")

# Move tensors and models to the GPU
tensor = torch.tensor([1, 2, 3])  # Create a tensor
tensor = tensor.to(device)  # Move tensor to the device (GPU or CPU)

