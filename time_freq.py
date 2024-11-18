import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg') 

fmin = 0.15  # Minimum frequency in Hz
fmax = 134.0  # Maximum frequency in Hz


def lectura(fif_file,duracion,overlap=3,save=False):
    raw_data=mne.io.read_raw_fif("SR_10min_cleaned.fif",preload=True)
    raw_data.crop(4*60,12*60)
    epochs = mne.make_fixed_length_epochs(raw_data, duration=duracion, overlap=overlap)#preload=True
    if save:
        output_epochs_file = 'Data/full_recording_epochs-epo.fif'
        epochs.save(output_epochs_file, overwrite=True)
    return epochs

def time_freq(epochs,picks,f_min,f_max):
    """
    Calcula epochs y transformación tiempo frecuencia
    Si no se indican picks, toma todos los canales.
    """
    if picks==None:
        picks=epochs.ch_names
        print(picks)
    # print(epochs.picks)
    frequencies=np.linspace(f_min,f_max,2*int(f_max-f_min))
    power = epochs.compute_tfr(
        method="morlet",
        picks=picks,
        freqs=frequencies,
        n_cycles=frequencies/2, #n_ciclos por cada freq
        #time_bandwidth=time_bandwidth,
        return_itc=False,
        #average=True,
        decim=3,
        n_jobs=5)
    
    return power

def proceso(fif_file,duracion,picks,f_min,f_max,overlap=3):
    epochs=lectura(fif_file,duracion,overlap=overlap)
    power=time_freq(epochs,picks,f_min,f_max)
    return epochs,power



