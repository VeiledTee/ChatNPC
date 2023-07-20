import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from bilstm_training import load_txt_file_to_dataframe

# Load the Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # You can choose a different model if you prefer
multinli_train = load_txt_file_to_dataframe('train')
multinli_match = load_txt_file_to_dataframe('match')
multinli_mismatch = load_txt_file_to_dataframe('mismatch')

multinli_df = pd.concat([multinli_train, multinli_match, multinli_mismatch], ignore_index=True)

pair_x = [s.strip() for s in multinli_df['sentence1']]
pair_y = [s.strip() for s in multinli_df['sentence2']]

# sentences = [
# ("The sky is blue.", "The sky is red."),
# ("Dogs bark.", "Dogs meow."),
# ("I love chocolate.", "I hate chocolate."),
# ("She is a doctor.", "She is a chef."),
# ("The Earth is flat.", "The Earth is round."),
# ("Summer is hot.", "Summer is cold."),
# ("He plays the guitar.", "He plays the piano."),
# ("It's raining outside.", "It's sunny outside."),
# ("The cat is sleeping.", "The cat is running."),
# ("The book is long.", "The book is short."),
# ("Elephants are small.", "Elephants are big."),
# ("Pizza is tasty.", "Pizza is disgusting."),
# ("The movie is funny.", "The movie is boring."),
# ("The ocean is shallow.", "The ocean is deep."),
# ("She is tall.", "She is short."),
# ("Coffee wakes me up.", "Coffee makes me sleepy."),
# ("I enjoy exercising.", "I hate exercising."),
# ("The car is fast.", "The car is slow."),
# ("Winter is warm.", "Winter is cold."),
# ("The concert was great.", "The concert was awful."),
# ("The moon is bright.", "The stars twinkle."),
# ("Apples are fruits.", "Bananas are fruits."),
# ("Running is good exercise.", "Swimming is good exercise."),
# ("Soccer is popular.", "Basketball is popular."),
# ("He reads books.", "She reads books."),
# ("I like ice cream.", "You like ice cream."),
# ("The flowers are blooming.", "The trees are blooming."),
# ("Music is relaxing.", "Art is relaxing."),
# ("The phone is ringing.", "The doorbell is ringing."),
# ("The beach is sandy.", "The desert is sandy."),
# ("Birds fly.", "Fish swim."),
# ("She sings beautifully.", "He sings beautifully."),
# ("The coffee is hot.", "The tea is hot."),
# ("The car is blue.", "The house is blue."),
# ("The river is flowing.", "The wind is blowing."),
# ("It's summer.", "It's winter."),
# ("I have a dog.", "I have a cat."),
# ("He dances well.", "She dances well."),
# ("The computer is new.", "The TV is new."),
# ("The movie is long.", "The song is long."),
# # Add more pairs here...
# ]
#
# labels = [
#     # Contradiction pairs (labeled as 1)
#     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#
#     # Non-contradiction pairs (labeled as 0)
#     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#     # Add more labels here...
# ]
# sentences.extend([(x, y) for x, y in zip(pair_x, pair_y)])
# labels.extend([1 if y == 'contradiction' else 0 for y in multinli_df['gold_label']])

# train_sentences = [(x, y) for x, y in zip(pair_x, pair_y)]
# train_labels = [1 if y == 'contradiction' else 0 for y in multinli_df['gold_label']]

sentences = [(x, y) for x, y in zip(pair_x, pair_y)]
labels = [1 if y == 'contradiction' else 0 for y in multinli_df['gold_label']]

# Generate embeddings for sentence pairs
embeddings = model.encode(sentences)
print(embeddings.shape)
print(len(labels))

# X_train = embeddings
# y_train = train_labels
# X_test = model.encode(test_sentences)
# y_test = test_labels

# Split the data into training and test sets, and get the indices
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2)

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
