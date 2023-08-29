document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('chat-form');
    const chatbox = document.getElementById('chatbox');
    const userInput = document.getElementById('user-input');
    const characterSelect = document.getElementById('character-select');
    console.log(characterSelect)
    console.log(userInput)

    // Add an event listener to the character select dropdown
    characterSelect.addEventListener('change', function () {
        console.log("Character selected:", characterSelect.value);

        if (characterSelect.value !== '') {
            // Enable the form elements when a character is selected
            userInput.disabled = false;
            form.querySelector('input[type="submit"]').disabled = false;
        } else {
            // Disable the form elements if no character is selected
            userInput.disabled = true;
            form.querySelector('input[type="submit"]').disabled = true;
        }
    });


    form.addEventListener('submit', function (e) {
        e.preventDefault(); // Prevent the form from submitting the traditional way

        const userMessage = userInput.value;

        // Append the user's message to the chatbox
        chatbox.innerHTML += `<p>User: ${userMessage}</p>`;

        // Clear the input field
        userInput.value = '';

        // Send the user's message to the server for processing
        fetch('/chat', {
            method: 'POST',
            body: JSON.stringify({ user_input: userMessage }),
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.text())
        .then(data => {
            // Append the chatbot's response to the chatbox
            chatbox.innerHTML += `<p>Chatbot: ${data}</p>`;

            // Scroll to the bottom of the chatbox
            chatbox.scrollTop = chatbox.scrollHeight;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
});
