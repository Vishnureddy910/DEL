import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Load + preprocess
X, y = load_iris(return_X_y=True)
y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

# Base model
def create_model():
    return Sequential([
        Dense(64, activation='relu', input_shape=(4,)),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])

# Train GD & SGD
def train(optimizer, batch):
    model = create_model()
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model.fit(X_train, y_train, epochs=50, batch_size=batch,
                     validation_data=(X_test, y_test), verbose=0)

gd_hist  = train(SGD(0.01), batch=32)
sgd_hist = train(SGD(0.01), batch=1)

# Plot helper
def plot_hist(h, title):
    plt.figure(figsize=(10,5))
    plt.plot(h.history['loss'], label='Train Loss')
    plt.plot(h.history['val_loss'], label='Val Loss')
    plt.title(f"{title} Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(); plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(h.history['accuracy'], label='Train Acc')
    plt.plot(h.history['val_accuracy'], label='Val Acc')
    plt.title(f"{title} Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid(); plt.show()

# Complete Plots
plot_hist(gd_hist, "GD")
plot_hist(sgd_hist, "SGD")
