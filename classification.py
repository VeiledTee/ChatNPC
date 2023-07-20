
    from sentence_transformers import SentenceTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report

    # Load the Sentence-BERT model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # You can choose a different model if you prefer

    # Sample list of sentences (replace this with your actual data)
    sentences = [
    # Contradiction pairs
    ("The sky is blue.", "The sky is red."),
    ("Dogs bark.", "Dogs meow."),
    ("I love chocolate.", "I hate chocolate."),
    ("She is a doctor.", "She is a chef."),
    ("The Earth is flat.", "The Earth is round."),
    ("Summer is hot.", "Summer is cold."),
    ("He plays the guitar.", "He plays the piano."),
    ("It's raining outside.", "It's sunny outside."),
    ("The cat is sleeping.", "The cat is running."),
    ("The book is long.", "The book is short."),
    ("Elephants are small.", "Elephants are big."),
    ("Pizza is tasty.", "Pizza is disgusting."),
    ("The movie is funny.", "The movie is boring."),
    ("The ocean is shallow.", "The ocean is deep."),
    ("She is tall.", "She is short."),
    ("Coffee wakes me up.", "Coffee makes me sleepy."),
    ("I enjoy exercising.", "I hate exercising."),
    ("The car is fast.", "The car is slow."),
    ("Winter is warm.", "Winter is cold."),
    ("The concert was great.", "The concert was awful."),

    # Non-contradiction pairs
    ("The moon is bright.", "The stars twinkle."),
    ("Apples are fruits.", "Bananas are fruits."),
    ("Running is good exercise.", "Swimming is good exercise."),
    ("Soccer is popular.", "Basketball is popular."),
    ("He reads books.", "She reads books."),
    ("I like ice cream.", "You like ice cream."),
    ("The flowers are blooming.", "The trees are blooming."),
    ("Music is relaxing.", "Art is relaxing."),
    ("The phone is ringing.", "The doorbell is ringing."),
    ("The beach is sandy.", "The desert is sandy."),
    ("Birds fly.", "Fish swim."),
    ("She sings beautifully.", "He sings beautifully."),
    ("The coffee is hot.", "The tea is hot."),
    ("The car is blue.", "The house is blue."),
    ("The river is flowing.", "The wind is blowing."),
    ("It's summer.", "It's winter."),
    ("I have a dog.", "I have a cat."),
    ("He dances well.", "She dances well."),
    ("The computer is new.", "The TV is new."),
    ("The movie is long.", "The song is long."),
    # Add more pairs here...
]

    labels = [
        # Contradiction pairs (labeled as 1)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,

        # Non-contradiction pairs (labeled as 0)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # Add more labels here...
    ]

    # Generate embeddings for sentence pairs
    embeddings = model.encode(sentences)

    # Split the data into training and test sets, and get the indices
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(embeddings, labels,
                                                                                     range(len(sentences)),
                                                                                     test_size=0.2)

    print("Training phrases:", [sentences[i] for i in train_indices])
    print("Test phrases:", [sentences[i] for i in test_indices])

    # Train the SVM classifier
    svm_classifier = SVC(kernel='linear', C=1.0)
    svm_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm_classifier.predict(X_test)

    # Evaluate the SVM classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)