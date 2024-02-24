from django.shortcuts import render, redirect
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
from keras.models import load_model
import openai
from django.core.mail import send_mail
from django.shortcuts import render
from .utils import send_prediction_email
import json
import random



openai.api_key = "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"


def generate_plot(eeg_data):
    # Filter data to get alpha (8-12 Hz) and beta (12-30 Hz) bands
    b_alpha, a_alpha = butter(4, [8/60, 12/60], btype='bandpass')
    b_beta, a_beta = butter(4, [12/60, 30/60], btype='bandpass')

    alpha_waves = lfilter(b_alpha, a_alpha, eeg_data)
    beta_waves = lfilter(b_beta, a_beta, eeg_data)

    # Plot the EEG signals
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True)
    axs[0].plot(eeg_data)
    axs[0].set_title('Raw EEG')
    axs[1].plot(alpha_waves, label='Alpha Waves (8-12 Hz)')
    axs[1].plot(beta_waves, label='Beta Waves (12-30 Hz)')
    axs[1].set_title('Filtered EEG')
    axs[1].legend()

    # Convert plot to PNG image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)

    # Encode PNG image to base64
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return plot_data

def eeg_view(request):
    # Load EEG data from file (data.npy)
    eeg_data = np.load('EEG/data/data.npy')
    import matplotlib
    matplotlib.use('agg')

    # Generate plot in a separate thread to avoid Matplotlib GUI warning
    plot_data = generate_plot(eeg_data)

    return render(request, 'eeg.html', {'plot_data': plot_data})

from django.shortcuts import render
from django.http import HttpResponse
from .forms import UploadFileForm

def upload_file(request):
    print("53")
    if request.method == 'POST':
        print("55")
        form = UploadFileForm(request.POST, request.FILES)
        print("56")
        csv_file = request.FILES['csvFile']
        # Pass the csv_file to your prediction function
        result = predictionEEG(csv_file)
        # Do something with the result, e.g., save it to a database or return it as a response
        #return HttpResponse(result)
        return render(request, 'output.html', {'result': result})
    else:
        print("64")
        form = UploadFileForm()
    return render(request, 'index.html', {'form': form})



def predictionEEG(filepath):
    print("68")
    df = pd.read_csv(filepath)
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

    return emotions[0]

def suggestionsChatGPT(request):
    age = '35'
    gender = 'Male'
    prompt = "give me suggestions to "+gender+" employee of age "+age+" to get rid of mental illness or mental stress while working, as key value pair like- Text:(4 point text),Audio:(names of 4 relaxing audio to get rid of mental stress),Video:(name of 4 videos to get rid of mental stress"
    print(prompt)
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": prompt}
        ])

    message = response.choices[0]['message']
    message = message['content']
    text = message.split('Audio')[0]
    audio = message.split('Audio')[1].split('Video')[0]
    video = message.split('Audio')[1].split('Video')[1]
    print("text: "+text)
    print("Audio: "+audio)
    print("video: "+video)
    return render(request, 'suggestions.html', {'text': text, 'audio':audio, 'video':video})

def sendPersonalizedEmail(request):
    employeeName = 'Chintamani'
    #age = '35'
    #gender = 'Male'
    employeeName = 'Chintamani'
    prompt = "draft me personalised email to send as a report of mental wellness of employee with name whos is receieint "+employeeName+" in json format with keys Subject and body with recepient name, email body will have that employee's mental health is not good to work and sugestion to deal with mental illness"
    print(prompt)
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": prompt}
        ])

    output = response.choices[0]['message']['content']
    data_dict = json.loads(output)
    print(data_dict)
    print(type(data_dict))
    subject = data_dict.get('Subject')
    body = data_dict.get('Body')
    print(data_dict.get('Subject'))
    print(data_dict.get('Body'))
    
    send_mail(
    subject,
    body,
    "hackhivesemicolon@gmail.com",
    ["chintamanisecond@gmail.com"],
    fail_silently=False,
    )
    return render(request, 'suggestions.html', {'subject': subject, 'body':body})

def predictionECG(filePath):
    df = pd.read_csv(filePath, header=None)
    ecgModel = load_model('EEG\static\ecgModel.h5')
    predict = ecgModel.predict(df)
    class_ = predict.argmax(axis=1)
    prob = (predict[0][class_]*100)-random.randint(4,11)
    prob = round(prob[0], 2)
    if class_ == 0:
        emotion = 'Angry'
    elif class_ == 1:
        emotion = 'Happy'
    elif class_ == '2':
        emotion = 'Sad'
    elif class_ == 3:
        emotion = 'Fear'
    elif class_ == 4:
        emotion = 'Surprized'
    print(prob)
    print(class_)
    print(emotion)
    return(emotion)

def upload_ecgfile(request):
    print("53")
    if request.method == 'POST':
        print("55")
        form = UploadFileForm(request.POST, request.FILES)
        print("56")
        csv_file = request.FILES['csvFile']
        # Pass the csv_file to your prediction function
        result = predictionECG(csv_file)
        # Do something with the result, e.g., save it to a database or return it as a response
        #return HttpResponse(result)
        return render(request, 'ecgOutput.html', {'class_': result})
    else:
        print("64")
        form = UploadFileForm()
    return render(request, 'index.html', {'form': form})


    

def send_email(request):
    send_prediction_email()
    return redirect('/')

def index(request):
    return render(request, 'index.html')
# views.py
from django.http import HttpResponse

def input_form(request):
    return render(request, 'input_form.html')

def process_input(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        age = request.POST.get('age')
        gender = request.POST.get('gender')
        email = request.POST.get('email')
        
        # Print the input values
        print(f"Name: {name}, Age: {age}, Gender: {gender}, Email: {email}")
        return HttpResponse("Input received successfully.")
    else:
        return HttpResponse("Invalid request method.")
