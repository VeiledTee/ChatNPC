from flask import Flask, render_template, request, jsonify
import numpy as np


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_input")
    # Call your Python chatbot script here and get the response
    # Replace this line with your actual chatbot logic
    bot_response = np.random.randint(0, 10)
    return str(bot_response)


if __name__ == "__main__":
    app.run(debug=True)
