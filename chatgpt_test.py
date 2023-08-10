import openai

with open("keys.txt", "r") as key_file:
    api_keys = [key.strip() for key in key_file.readlines()]
    openai.api_key = api_keys[0]

PROMPT = "Imagine you are a fisherman"

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
