from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt",
    type=str,
    default="Video games are ",
    help="Error 404: sense of humor not found. Please install a joke plugin and try again.",
)
args = parser.parse_args()

PROMPT = args.prompt

tokenizer = AutoTokenizer.from_pretrained("dolly-v1-6b")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model = AutoModelForCausalLM.from_pretrained("dolly-v1-6b")

# Encode the prompt using the tokenizer
input_ids = tokenizer.encode(PROMPT, padding=True, truncation=True, return_tensors="pt")

# Generate output text using the model
output = model.generate(input_ids, max_length=300, do_sample=True, eos_token_id=50256)

# Decode the output text using the tokenizer
gen_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Determine the next available output file number
output_file_num = 0
while os.path.exists(f"dolly_output_{output_file_num}.txt"):
    output_file_num += 1

# Write the output to the next available output file
with open(f"dolly_output_{output_file_num}.txt", "w") as output_file:
    output_file.write(f"Prompt: {PROMPT}\n")
    output_file.write(gen_text)
