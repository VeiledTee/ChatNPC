from flask import Flask, render_template, request, jsonify
import numpy as np

import webchat

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat() -> str:
    user_input: str = request.json.get("user_input")  # what user said
    selected_character: str = request.json.get("character_select")  # character name
    reply: str = webchat.run_query_and_generate_answer(query=user_input, receiver=selected_character)
    return f"{reply}"


@app.route('/upload_background', methods=['POST'])
def upload_background() -> str:
    data = request.get_json()
    selected_character = data.get('character')

    webchat.upload_background(selected_character)

    print("Background uploaded successfully")
    return ''


if __name__ == "__main__":
    app.run(debug=True)
