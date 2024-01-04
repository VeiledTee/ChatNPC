from openai import OpenAI

with open("keys.txt", "r") as key_file:
    api_keys = [key.strip() for key in key_file.readlines()]
    client = OpenAI(api_key=api_keys[0])

TEXT_MODEL: str = "gpt-4-1106-preview"


def find_importance(fact: str) -> int:
    prompt_template = "Prompts/poignancy.txt"
    # prompt_input = create_prompt_input(persona, event_description)
    prompt = ("Here is a brief description of Peter Satoru. "
              "Peter Satoru is a respected and experienced archer in the town of Ashbourne. He has lived there his entire life, and at 65 years old, he still possesses excellent archery skills and a wealth of knowledge about combat and strategy. Despite his age, Peter is patient, resilient, and wise, making him a valuable asset to the community. He spends much of his time training younger archers in the town, passing on his expertise to the next generation. Peter's family background includes a father who was also an archer, which suggests a long tradition of archery in his family. His social status is lower class, but his allegiances lie with the people of Ashbourne, whom he serves and protects. Peter has formed close relationships with Jack McCaster, a local fisherman, and Melinda Deek, a fellow knight in the village. "
              "On the scale of 1 to 10, "
              "where 1 is purely mundane (e.g., I need to do the dishes, I need to walk the dog) "
              "and 10 is extremely significant (e.g., I wish to become a professor, I love Elie), "
              "rate the likely significance of the following information to Peter Satoru. Information: "
              "Jack McCaster's favourite fish is cod."
              "Rate (return a number between 1 to 10):")

    example_output = "5"  ########
    special_instruction = "ONLY include ONE integer value on the scale of 1 to 10 as the output."

    prompt += f"{special_instruction}\n"
    prompt += "Example output integer:\n"
    prompt += str(example_output)

    res = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        top_p=1,
        max_tokens=1,
        stream=False,
        presence_penalty=0,
        frequency_penalty=0,
    )  # conversation with LLM
    clean_res: str = str(res.choices[0].message.content).strip()  # get model response


    print(clean_res)

find_importance('')

