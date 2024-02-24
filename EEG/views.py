from django.shortcuts import render
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
import io
import base64


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
