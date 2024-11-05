import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg') 

fmin = 0.15  # Minimum frequency in Hz
fmax = 134.0  # Maximum frequency in Hz


def lectura(fif_file):
    raw_data=mne.io.read_raw_fif("SR_10min_cleaned.fif",preload=True)
    raw_data.crop(4*60,12*60)
    return raw_data

def time_freq(raw_data,duracion,picks,f_min,f_max):
    if picks==None:
        picks=raw_data.ch_names
        print(picks)
    epochs = mne.make_fixed_length_epochs(raw_data, duration=duracion, overlap=0.5)#preload=True
    print(epochs.picks)
    frequencies=np.linspace(f_min,f_max,2*int(f_max-f_min))
    power = epochs.compute_tfr(
        method="morlet",
        picks=picks,
        freqs=frequencies,
        n_cycles=frequencies/2, #n_ciclos por cada freq
        #time_bandwidth=time_bandwidth,
        return_itc=False,
        #average=True,
        n_jobs=5)
    
    return epochs,power

def proceso(fif_file,duracion,picks,f_min,f_max):
    raw_data=lectura(fif_file)
    epochs,power=time_freq(raw_data,duracion,picks,f_min,f_max)
    return epochs,power
