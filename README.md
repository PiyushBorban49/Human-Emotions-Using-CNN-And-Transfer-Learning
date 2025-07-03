# 🎧 Human Emotion Detection from Voice using CNN & Transfer Learning (ResNet152V2)

This project detects human emotions from voice recordings using **Mel spectrograms** and **ResNet152V2** (Transfer Learning). It classifies audio into 8 emotional states: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprised.

---

## 📂 Dataset

We use the [RAVDESS Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio), containing emotional speech by 24 professional actors.

Folder structure:
audio_speech_actors_01-24/
├── Actor_01/
│ ├── 03-01-01-01-01-01-01.wav
│ ├── ...
├── Actor_02/
│ ├── ...
...

yaml
Copy
Edit

---

## 😃 Emotion Labels

| Code | Emotion   |
|------|-----------|
| 01   | Neutral   |
| 02   | Calm      |
| 03   | Happy     |
| 04   | Sad       |
| 05   | Angry     |
| 06   | Fearful   |
| 07   | Disgust   |
| 08   | Surprised |

---

## 🧠 How it Works

### 🎼 Preprocessing:
- Load `.wav` files using `librosa`
- Pad/truncate to 3 seconds (22050×3 samples)
- Convert to Mel Spectrogram
- Normalize & resize to 224×224×3
- Stack spectrogram into 3 channels (RGB style)

### 🧱 Model Architecture:
```python
ResNet152V2 (imagenet, frozen)
→ Flatten
→ Dense(128, relu) + BatchNorm + Dropout(0.5)
→ Dense(64, relu) + BatchNorm + Dropout(0.3)
→ Dense(32, relu) + BatchNorm + Dropout(0.2)
→ Dense(8, softmax)
🏋️‍♂️ Training
Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Batch size: 32

Epochs: 5

EarlyStopping and ReduceLROnPlateau used for better generalization

Training:

python
Copy
Edit
model.fit(X_train, Y_train,
          validation_data=(X_test, Y_test),
          epochs=5,
          batch_size=32,
          callbacks=[early_stop, reduce_lr])
📊 Results
Basic training/validation loss plots are shown to monitor learning:

python
Copy
Edit
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
💾 Model Saving
The model is saved as:

Copy
Edit
emotion_detection_resnet_model.keras
✅ Tracked using Git LFS (Large File Storage) since it's ~371 MB.

🚀 How to Clone
Install Git LFS:

bash
Copy
Edit
git lfs install
Clone the repository:

bash
Copy
Edit
git clone https://github.com/PiyushBorban49/Human-Emotions-Using-CNN-And-Transfer-Learning.git
Download the dataset from Kaggle
Place it in:

bash
Copy
Edit
./audio_speech_actors_01-24/
🧰 Requirements
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
requirements.txt (sample):
txt
Copy
Edit
tensorflow
keras
numpy
pandas
opencv-python
librosa
matplotlib
scikit-learn
📁 Project Structure
bash
Copy
Edit
├── model.py                    # Training & model code
├── emotion_detection_resnet_model.keras  # Trained model (LFS)
├── audio_speech_actors_01-24/ # Dataset folder
├── README.md
├── .gitattributes
├── .gitignore
└── requirements.txt
👨‍💻 Author
Piyush Borban
GitHub: @PiyushBorban49

📜 License
This project is licensed under the MIT License.
