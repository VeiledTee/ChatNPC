from flask import Flask, render_template, request, jsonify
import numpy as np

import webchat


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_input")
    selected_character = request.json.get("character_select")
    # Call your Python chatbot script here and get the response
    # Replace this line with your actual chatbot logic
    return f"{user_input} | {selected_character} | {webchat.random_sentence()}"


if __name__ == "__main__":
    app.run(debug=True)
