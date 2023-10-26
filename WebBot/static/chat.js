document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('chat-form');
    const chatbox = document.getElementById('chatbox');
    const userInput = document.getElementById('user-input');
    const characterSelect = document.getElementById('character-select');

    let selectedCharacter = ''; // Initialize the selected character name

    // Add an event listener to the character select dropdown
    characterSelect.addEventListener('change', function () {
        selectedCharacter = characterSelect.options[characterSelect.selectedIndex].text;
        if (characterSelect.value !== '') {
            // Enable the form elements when a character is selected
            userInput.disabled = false;
            form.querySelector('input[type="submit"]').disabled = false;

            // Make an HTTP request to call the Python function
            fetch('/upload_background', {
                method: 'POST',
                body: JSON.stringify({ character: selectedCharacter }),
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => {
                if (response.ok) {
                    // Handle a successful response here
                    console.log("Background uploaded successfully");
                } else {
                    // Handle an error response here
                    console.error("Error uploading background");
                }
            })
            .catch(error => {
                // Handle any network or other errors here
                console.error("Network or other error occurred");
            });
        } else {
            // Disable the form elements if no character is selected
            userInput.disabled = true;
            form.querySelector('input[type="submit"]').disabled = true;
        }
    });

    form.addEventListener('submit', function (e) {
        e.preventDefault(); // Prevent the form from submitting the traditional way

        const userMessage = userInput.value;

        // Append the user's message to the chatbox with the Player label
        chatbox.innerHTML += `<p><strong>Player:</strong> ${userMessage}</p>`;

        // Clear the input field
        userInput.value = '';

        // Send the user's message to the server for processing
        fetch('/chat', {
            method: 'POST',
            body: JSON.stringify({ user_input: userMessage, character_select: selectedCharacter }), // Include selected character
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.text())
        .then(data => {
            // Append the character's name and response to the chatbox
            chatbox.innerHTML += `<p><strong>${selectedCharacter}:</strong> ${data}</p>`;

            // Scroll to the bottom of the chatbox
            chatbox.scrollTop = chatbox.scrollHeight;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
});
