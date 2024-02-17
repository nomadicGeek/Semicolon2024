import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Load EEG data
eeg_data = np.load('data.npy') 

# Filter data to get alpha (8-12 Hz) and beta (12-30 Hz) bands
b_alpha, a_alpha = butter(4, [8/60, 12/60], btype='bandpass') 
b_beta, a_beta = butter(4, [12/60, 30/60], btype='bandpass')

alpha_waves = lfilter(b_alpha, a_alpha, eeg_data)
beta_waves = lfilter(b_beta, a_beta, eeg_data)

# Plot the EEG signals
plt.subplot(2, 1, 1)
plt.plot(eeg_data)
plt.title('Raw EEG')

plt.subplot(2, 1, 2)
plt.plot(alpha_waves)
plt.title('Alpha Waves (8-12 Hz)')
plt.plot(beta_waves)
plt.title('Beta Waves (12-30 Hz)')

plt.tight_layout()
plt.show()
