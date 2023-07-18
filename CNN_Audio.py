### =====================================================================================================================
### Install the required Python libraries through CMD prompt using this command:
### pip install tensorflow librosa numpy matplotlib
### =====================================================================================================================

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, utils

# Function to extract spectrograms from audio files
def extract_spectrogram(file_path, n_mels=128, n_fft=2048, hop_length=512):
    audio, sr = librosa.load(file_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

# Function to load audio files and extract spectrograms with labels
def load_data(data_dir):
    spectrograms = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        for audio_file in os.listdir(label_dir):
            file_path = os.path.join(label_dir, audio_file)
            spectrogram = extract_spectrogram(file_path)
            spectrograms.append(spectrogram)
            labels.append(label)
    return np.array(spectrograms), np.array(labels)

# Load the audio data and labels
data_dir = 'path/to/your/audio/dataset'
spectrograms, labels = load_data(data_dir)

# Encode labels as integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(spectrograms, encoded_labels, test_size=0.2, random_state=42)

# Reshape the data to add the channel dimension (for CNN)
X_train = X_train.reshape((*X_train.shape, 1))
X_test = X_test.reshape((*X_test.shape, 1))

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Plot the training and validation accuracy over epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

### ==================================================================================================================================================================================
### This code will load your audio dataset, create spectrograms from the audio files, build a simple CNN model, and then train and evaluate the model. 
### Adjust the hyperparameters of the CNN layers and other parameters as needed to suit your specific dataset and requirements. 
### Also, ensure that your dataset is organized in separate folders for each class with audio files in the appropriate formats (e.g., WAV, MP3) for this code to work as expected.
### ==================================================================================================================================================================================
