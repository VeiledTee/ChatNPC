import glob
import os
from time import perf_counter
import logging

from flask import Flask, render_template, jsonify, request, session, send_from_directory, Response

import global_functions
import webchat
from global_functions import get_network_usage
from keys import flask_secret_key

start_sent, start_received = get_network_usage()

# configure flask app
app = Flask(__name__)
app.static_folder = "static"
app.secret_key = flask_secret_key

# configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat() -> Response:
    user_input: str = request.json.get("user_input")  # what the user said

    if user_input.lower() == "bye":
        end_sent, end_recv = get_network_usage()

        # Calculate the difference in bytes
        bytes_sent = end_sent - start_sent
        bytes_recv = end_recv - start_received

        logger.info(f"Sent: {bytes_sent / (1024 * 1024):.2f} MB")  # convert to MB
        logger.info(f"Received: {bytes_recv / (1024 * 1024):.2f} MB")  # convert to MB
        return "Goodbye!"

    selected_character: str = request.json.get("character_select")  # character name

    time_start = perf_counter()
    cur_namespace: str = global_functions.name_conversion(to_snake=True, to_convert=selected_character)

    context: list[str] = webchat.retrieve_context_list(
        namespace=cur_namespace,
        query=user_input,
        impact_score=True,
    )

    logger.info(f"Pre-check context: \t{context}")

    base_options: list[str] = ["Both statements are true", "Neither statement is true"]
    contradictory_premises = None
    contradiction: bool = False

    for premise in context:
        if webchat.are_contradiction(premise_a=user_input, premise_b=premise):
            contradictory_premises = [premise, user_input]
            webchat.upload_contradiction(
                namespace=cur_namespace,
                s1=premise,
                s2=user_input,
            )
            contradiction = True
            break

    # webchat.upload_contradiction(
    #     namespace=cur_namespace,
    #     s1="test 3",
    #     s2="user query",
    # )

    logger.info(
        f"User Input: \t\t\t{user_input}")
    logger.info(
        f"Contradiction: \t\t\t{contradiction}")
    logger.info(
        f"Contradictory Premises: {contradictory_premises}")

    if contradiction:
        # Define options when the flag is encountered
        options = [contradictory_premises[0], contradictory_premises[1]] + base_options
        logger.info(f"Options: {options}")
        session["options"] = options  # Store options in session for future reference
        response_text = "Which of the following statements is true?"
        return jsonify({"response": response_text, "options": options})

    selected_option = request.json.get("selected_option", None)
    options = session.get("options", [])  # Retrieve options from session

    if selected_option == 0:  # DB is correct, new info is wrong
        logger.info(f"Context: {context}")
        logger.info(f"Selected Option: {selected_option} | {options[selected_option]}")
        webchat.handle_contradiction(
            contradictory_index=selected_option,
            namespace=cur_namespace,
        )
    elif selected_option == 1:  # DB is incorrect, new info is correct
        logger.info(f"Context: {context}")
        context[context.index(options[0])] = options[selected_option]
        logger.info(f"Selected Option: {selected_option} | {options[selected_option]}")
        webchat.handle_contradiction(
            contradictory_index=selected_option,
            namespace=cur_namespace,
        )
        logger.info(f"Updated Context: {context}")
    elif selected_option == 2:  # both correct
        logger.info(f"Context: {context}")
        context.append(options[1])
        logger.info(f"Selected Option: {selected_option} | {options[selected_option]}")
        logger.info(f"Updated Context: {context}")
        webchat.handle_contradiction(
            contradictory_index=selected_option,
            namespace=cur_namespace,
        )
    elif selected_option == 3:  # neither correct
        logger.info(f"Context: {context}")
        context.remove(options[0])
        logger.info(f"Selected Option: {selected_option} | {options[selected_option]}")
        logger.info(f"Updated Context: {context}")
        webchat.handle_contradiction(
            contradictory_index=selected_option,
            namespace=cur_namespace,
        )

    logger.info(f"Context: {context}")

    reply, prompt_tokens, reply_tokens = webchat.run_query_and_generate_answer(
        query=user_input, receiver=selected_character, context=context
    )

    time_end = perf_counter()
    time_difference = time_end - time_start
    time_passed_per_prompt_token = time_difference / prompt_tokens
    time_passed_per_reply_token = time_difference / reply_tokens

    logger.info(f"Time taken to reply: {time_difference:.2f} seconds")
    logger.info(f"Time per prompt token: {time_passed_per_prompt_token:.4f} seconds/token")
    logger.info(f"Time per reply token: {time_passed_per_reply_token:.4f} seconds/token")

    return jsonify({"character": selected_character, "response": reply, "selected_option": None})


@app.route("/upload_background", methods=["POST"])
def upload_background() -> str:
    data = request.get_json()
    selected_character = data.get("character")
    logger.info("Backgrounding")
    webchat.upload_background(selected_character)
    logger.info(f"Background uploaded successfully for {selected_character}")
    return ""


@app.route("/get_latest_audio/<character_name>")
def get_latest_audio(character_name):
    audio_dir = os.path.join(os.path.join("static", "audio"), character_name)

    # Get a list of all audio files in the directory
    audio_files = glob.glob(os.path.join(audio_dir, "*.mp3"))

    # Sort the files by modification time (most recent first)
    audio_files.sort(key=os.path.getmtime, reverse=True)

    # Get the URL of the most recent audio file
    if audio_files:
        latest_audio_filename = os.path.basename(audio_files[0])
        latest_audio_url = f"/get_audio/{character_name}/{latest_audio_filename}"
        return jsonify({"latest_audio_url": latest_audio_url})
    else:
        return jsonify({"error": "No audio files found"})


@app.route("/get_audio/<character_name>/<filename>")
def get_audio(character_name, filename):
    audio_dir = os.path.join(os.path.join("static", "audio"), character_name)
    return send_from_directory(audio_dir, filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
