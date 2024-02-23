import pandas as pd
from keras.models import load_model

df = pd.read_csv('EEG\static\emotions.csv')
df = df.drop('label', axis=1)
df = df.drop(index=df.index[0], axis=0)
model = load_model('EEG\static\model.h5')
predicted_emotion = model.predict(df)
emotions = []
for i in predicted_emotion:
    if i[0] == 1.0:
        emotions.append("Positive")
    elif i[1] == 1.0:
        emotions.append("Negative")
    elif i[2] == 1.0:
        emotions.append("Neutral")
print(emotions[0])