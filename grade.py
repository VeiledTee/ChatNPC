from chat import name_conversion


def calculate_grade(character: str) -> str:
    sol: str = load_solution(character)
    most_recent_sub: str = load_submission(character)
    print(sol, most_recent_sub)

    if len(sol) != len(most_recent_sub):
        raise ValueError("Strings must be the same length.")

    matching_count = 0
    total_chars = len(sol)

    for i in range(total_chars):
        if sol[i] == most_recent_sub[i]:
            matching_count += 1

    percentage = (matching_count / total_chars) * 100

    return f"{percentage:.2f}"


def load_solution(character: str) -> str:
    """
    Load the correct answers to the exam
    :param character: Character exam solutions to load
    :return: String of correct answers
    """
    path: str = f"Data/MC Solutions/{name_conversion(True, character)}_solutions.txt"
    with open(path, "r") as sol_file:
        solutions: str = ""
        for line in sol_file.readlines():
            solutions += line.strip()[-1]
    return solutions


def load_submission(character: str) -> str:
    """
    Load submitted answers to exam
    :param character: Character being graded
    :return: String of all answers given by LLM
    """
    path: str = f"Data/MC Results/{name_conversion(True, character)}_submissions.txt"
    with open(path, "r") as sub_file:
        submissions: str = ""
        for line in sub_file.readlines():
            submissions += line.strip()[-1]

    return submissions.split("=")[-2][:-1]


if __name__ == "__main__":
    # CHARACTER: str = "John Pebble"  # thief
    # CHARACTER: str = "Evelyn Stone-Brown"  # blacksmith
    CHARACTER: str = "Caleb Brown"  # baker
    # CHARACTER: str = 'Jack McCaster'  # fisherman
    # CHARACTER: str = "Peter Satoru"  # archer
    # CHARACTER: str = "Melinda Deek"  # knight
    # CHARACTER: str = "Sarah Ratengen"  # tavern owner

    print(calculate_grade(CHARACTER))
