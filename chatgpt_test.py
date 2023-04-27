import openai

openai.api_key = "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"
PROMPT = "Imagine you are a fisherman"

# response = openai.Completion.create(
#   model="ada",
#   prompt=PROMPT,
#   temperature=0.6
# )

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=0.6,
    messages=[{"role": "user", "content": PROMPT}],
)

with open("prompt_response.txt", "a") as file:
    file.write(
        f"Prompt: {PROMPT}\nResponse: {response['choices'][0]['message']['content']}========================================================================================================================"
    )

print(response)
