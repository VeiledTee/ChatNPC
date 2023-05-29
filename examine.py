import json
from typing import List

import openai

from chat import answer


def extract_data_multi_string(lines: List[str]) -> str:
    """
    Takes a list of strings and combines them into a multi-line string
    :param lines: a list of the lines
    :return: all lines as multiple strings
    """
    data: str = ""
    for line in lines:
        data += line.strip() + "\n"
    return data.strip()


def extract_data_single_string(lines: List[str]) -> str:
    """
    Takes a list of strings and combines them into a single string
    :param lines: a list of the lines
    :return: all lines as a single string
    """
    data: str = ""
    for line in lines:
        data += line.strip() + " "
    return data.strip()


def get_background(character_file: str) -> str:
    with open(character_file, "r") as read_file:
        return extract_data_single_string(read_file.readlines())


def get_town(town_name: str) -> str:
    with open(f"Text Summaries/{town_name.lower()}.txt", "r") as read_file:
        return extract_data_single_string(read_file.readlines())


def name_conversion(to_snake: bool, to_convert: str) -> str:
    if to_snake:
        text = to_convert.lower().split(" ")
        converted: str = text[0]
        for i, t in enumerate(text):
            if i == 0:
                pass
            else:
                converted += f"_{t}"
        return converted
    else:
        text = to_convert.split("_")
        converted: str = text[0].capitalize()
        for i, t in enumerate(text):
            if i == 0:
                pass
            else:
                converted += f" {t.capitalize()}"
        return converted


def write_exam(character: str, chat_history: List[dict]) -> str:
    character = name_conversion(
        to_snake=True, to_convert=character
    )  # get character name formatted correctly

    with open(
        f"Data/MC Tests/{character}_test.txt", "r"
    ) as exam_file:  # extract exam
        exam: str = extract_data_multi_string(exam_file.readlines())

    submission: str = answer(exam, chat_history)  # generate response

    with open(
        f"Data/MC Results/{character}_submissions.txt", "a"
    ) as answer_file:  # save responses
        answer_file.write(f"{submission}\n=====\n")

    return submission  # return responses


if __name__ == "__main__":
    HISTORY: List[dict] = []

    TOWN: str = "Ashbourne"

    # pick character
    # CHARACTER: str = "John Pebble"  # thief
    # CHARACTER: str = "Evelyn Stone-Brown"  # blacksmith
    CHARACTER: str = "Caleb Brown"  # baker
    # CHARACTER: str = 'Jack McCaster'  # fisherman
    # CHARACTER: str = "Peter Satoru"  # archer
    # CHARACTER: str = "Melinda Deek"  # knight
    # CHARACTER: str = "Sarah Ratengen"  # tavern owner

    with open("Text Summaries/characters.json", "r") as f:
        names = json.load(f)

    DATA_FILE: str = f"Text Summaries/{names[CHARACTER]}.txt"

    background_info: str = get_background(DATA_FILE)
    town_info: str = get_town(TOWN)

    HISTORY.append(
        {
            "role": "system",
            "content": f"Information about your hometown: {town_info}\n Information about you {background_info}",
        }
    )

    with open('keys.txt', 'r') as key_file:
        openai.api_key = (key_file.readlines()[0])

    print(write_exam(CHARACTER, HISTORY))

    """
    get character
    get background
    get town info
    use as system prompt
    ask questions 
    output answers in order of questions asked
    """
