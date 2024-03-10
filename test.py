import torch
from config import DEVICE, MODEL, TOKENIZER

premise = "Frank loves muffins"
hypothesis = "Frank's favourite food are muffins"

input = TOKENIZER(premise, hypothesis, truncation=True, return_tensors="pt").to(DEVICE)
output = MODEL(input["input_ids"].to(DEVICE))
prediction = torch.softmax(output["logits"][0], -1).tolist()
label_names = ["contradiction"]
prediction = {name: round(float(pred) * 100, 4) for pred, name in zip(prediction, label_names)}
prediction["non-contradiction"] = round(100 - prediction["contradiction"], 4)
print(prediction)
if prediction["contradiction"] >= prediction["non-contradiction"]:
    print(True)
else:
    print(False)
