import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import cv2
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import librosa
import librosa.display
import IPython.display as ipd
from alembic.command import history
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout,Flatten,Input
from keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# creating class to extract information from file name
class EmotionLabelEncoder():
    def __init__(self):
        self.emotion_object = {
            '01': 0,  # neutral
            '02': 1,  # calm
            '03': 2,  # happy
            '04': 3,  # sad
            '05': 4,  # angry
            '06': 5,  # fearful
            '07': 6,  # disgust
            '08': 7   # surprised
        }

        self.emotion_intensity = {
            "01" : "normal",
            "02" : "strong"
        }

    def fit(self,filename):
        temp = filename.replace('.wav','').split('-')
        return (self.emotion_object[temp[2]])



# preparing data
data = []
labels = []

le = EmotionLabelEncoder()
desired_length = 22050 * 3  # 3 seconds of audio

root_path = r"C:\Users\piyus\OneDrive\Desktop\python\artificial intelligence\Human Emotion through voice\audio_speech_actors_01-24"

for folder in os.listdir(root_path):
    folder_path = os.path.join(root_path, folder)
    for voice in os.listdir(folder_path):
        file_path = os.path.join(folder_path, voice)

        y, sr = librosa.load(file_path, sr=22050)
        y = librosa.util.fix_length(y, size=desired_length)

        S = librosa.feature.melspectrogram(y=y, sr = sr)
        S_DB = librosa.power_to_db(S, ref=np.max)
        S_DB = np.nan_to_num(S_DB, nan=0.0, posinf=0.0, neginf=0.0)

        img = cv2.resize(S_DB, (224, 224))
        img = np.stack((img,) * 3, axis=-1)

        data.append(img)
        labels.append(le.fit(voice))

data = np.array(data, dtype=float)/ 255.0
labels = np.array(labels)

X_train,X_test,Y_train,Y_test = train_test_split(data,labels,test_size=0.2,shuffle=True)

input_tensor = Input(shape=(224, 224, 3))
conv_base = ResNet152V2(weights='imagenet', include_top=False, input_tensor=input_tensor)
conv_base.trainable = False

x = conv_base.output
x = Flatten()(x)

x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

output = Dense(8, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output)

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

early_stop = EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(patience=3)

history = model.fit(X_train,Y_train,validation_data=(X_test,Y_test),verbose=1,epochs=5,batch_size=32,callbacks=[early_stop, reduce_lr])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()


model.save("emotion_detection_resnet_model.keras")
print("Model saved successfully!")
