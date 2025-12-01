import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Load data
max_features = 10000
maxlen = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Reverse word index for displaying samples
word_index = imdb.get_word_index()
idx_to_word = {i: w for w, i in word_index.items()}

print("\nIMDb Samples:\n")
for i in range(5):
    text = " ".join(idx_to_word.get(idx, "?") for idx in x_train[i])
    print(f"Review {i+1}: {text}\nSentiment: {y_train[i]}\n")

# Pad sequences
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Model
model = Sequential([
    Embedding(max_features, 50, input_length=maxlen),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

# Prediction helper
def predict_sentiment(text):
    idx = imdb.get_word_index()
    seq = [idx.get(w, 0) for w in text.split() if idx.get(w, 0) < max_features]
    seq = pad_sequences([seq], maxlen=maxlen)
    return model.predict(seq)[0][0]

print("Positive:", predict_sentiment("This movie was fantastic, I loved every moment!"))
print("Negative:", predict_sentiment("I couldn't stand this movie, it was terrible."))

# Plots
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Accuracy")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Loss")
plt.legend()
plt.show()
