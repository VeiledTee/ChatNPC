import glob
import os

from flask import Flask, render_template, jsonify, request, send_from_directory, Response

import webchat
from global_functions import get_network_usage
from time import time

start_sent, start_recv = get_network_usage()
app = Flask(__name__)
app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat() -> Response:
    user_input: str = request.json.get("user_input")  # what the user said

    if user_input.lower() == 'bye':
        end_sent, end_recv = get_network_usage()

        # Calculate the difference in bytes
        bytes_sent = end_sent - start_sent
        bytes_recv = end_recv - start_recv

        print(f"Sent: {bytes_sent / (1024 * 1024):.2f} MB")  # convert to MB
        print(f"Received: {bytes_recv / (1024 * 1024):.2f} MB")  # convert to MB
        return "Goodbye!"

    selected_character: str = request.json.get("character_select")  # character name

    time_start = time()

    # Check for the specific flag
    if user_input.lower() == 'flag':
        response_text = f"{selected_character}: Which of the following statements is true?"
        options = ["Option A", "Option B", "Option C", "Option D"]
        return jsonify({'character': selected_character, 'response': response_text, 'options': options})

    # If the flag is not detected, proceed with the regular response generation
    reply, prompt_tokens, reply_tokens = webchat.run_query_and_generate_answer(query=user_input,
                                                                               receiver=selected_character)

    time_end = time()

    time_difference = time_end - time_start
    time_passed_per_prompt_token = time_difference / prompt_tokens
    time_passed_per_reply_token = time_difference / reply_tokens

    print(f"Time taken to reply: {time_difference:.2f} seconds")
    print(f"Time per prompt token: {time_passed_per_prompt_token:.4f} seconds/token")
    print(f"Time per reply token: {time_passed_per_reply_token:.4f} seconds/token")

    # Include the character's name in the response
    response_text_with_name = f"{selected_character}: {reply}"

    return jsonify({'character': selected_character, 'response': response_text_with_name})


@app.route('/upload_background', methods=['POST'])
def upload_background() -> str:
    data = request.get_json()
    selected_character = data.get('character')
    print('Backgrounding')
    webchat.upload_background(selected_character)

    print(f"Background uploaded successfully for {selected_character}")
    return ''


@app.route('/get_latest_audio/<character_name>')
def get_latest_audio(character_name):
    audio_dir = os.path.join(os.path.join("static", "audio"), character_name)

    # Get a list of all audio files in the directory
    audio_files = glob.glob(os.path.join(audio_dir, '*.mp3'))

    # Sort the files by modification time (most recent first)
    audio_files.sort(key=os.path.getmtime, reverse=True)

    # Get the URL of the most recent audio file
    if audio_files:
        latest_audio_filename = os.path.basename(audio_files[0])
        latest_audio_url = f"/get_audio/{character_name}/{latest_audio_filename}"
        return jsonify({'latest_audio_url': latest_audio_url})
    else:
        return jsonify({'error': 'No audio files found'})


@app.route('/get_audio/<character_name>/<filename>')
def get_audio(character_name, filename):
    audio_dir = os.path.join(os.path.join("static", "audio"), character_name)
    return send_from_directory(audio_dir, filename)


if __name__ == "__main__":
    app.run(debug=True)
