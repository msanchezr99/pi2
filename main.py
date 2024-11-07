import os
from time_freq import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg') 
print(matplotlib.get_backend()) #"agg"
picks=["n'1", "n'2","n'3", "n'4", "cc'1", "cc'2", "cc'3", "cc'4", "cc'5", "m'1", "m'2"]
#  ,"m'3", "m'4", "m'5", "sc'2", "sc'3", "sc'4", "sc'5", "sc'6", "lp'1", "lp'2", "lp'3", "lp'4"
#  ,"lp'5", "y'1", "y'2", "y'3", "y'4", "y'5", "y'6", "oc'1", "oc'2", "oc'3", "oc'4", "op'1", "op'2",
#  "op'3", "op'4", "pi'1", "pi'2", "pi'3", "pa'1", "pa'2", "pa'3", "pa'4"]


def save_images(data):
    output_dir = 'output_images'
    os.makedirs(output_dir, exist_ok=True)

    # Loop over each epoch and channel
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            # Extract the 2D slice
            image_data = data[epoch_idx, ch_idx, :, :]
            
            plt.figure()
            plt.imshow(image_data, aspect='auto', origin='lower', cmap='viridis')  # Customize cmap as needed
            plt.colorbar()
            plt.title(f'Epoch {epoch_idx}, Channel {ch_idx}')
            
            # Save the figure
            filename = f'epoch_{epoch_idx}_channel_{ch_idx}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            
            # Close the plot to free memory
            plt.close()

if __name__=="__main__":
    f_min=30
    f_max=100
    epochs,power=proceso("SR_10min_cleaned.fif",30,picks,f_min,f_max)
    n_epochs,n_channels,n_freqs,n_times=power.data.shape
    print(f"n_times: {n_times}, {n_freqs}")
    epoch_idx=8
    # ch_idx=6
    save_images(power.data)
    # for epoch_idx in range(12):
    #     for ch_idx in [0,2,6]:
    #         spectrogram = np.log(power.data[epoch_idx, ch_idx, :, :])

    #         #epochs.compute_psd().plot(amplitude=False)
    #         print("Epochs extent: ",epochs.times[0], epochs.times[-1], f_min, f_max)
    #         # Plot spectrogram
    #         plt.figure(figsize=(10, 5))
    #         plt.imshow(spectrogram, origin='lower',extent=[epochs.times[0], epochs.times[-1], f_min, f_max])
    #         plt.colorbar(label='Power')
    #         plt.title(f'Epoch {epoch_idx}, Channel {ch_idx }')
    #         plt.xlabel('Time (s)')
    #         plt.ylabel('Frequency (Hz)')
    #         plt.show()

        



    # for i, epoch_power_data in enumerate(all_epoch_powers[:3]):  # Plot the first 3 epochs
    #     plt.figure(figsize=(10, 5))
    #     plt.imshow(
    #         epoch_power_data.mean(axis=0), aspect='auto', origin='lower',
    #         extent=[0, epoch_power_data.shape[2], gamma_frequencies[0], gamma_frequencies[-1]]
    #     )
    #     plt.colorbar(label='Power')
    #     plt.xlabel('Time (downsampled)')
    #     plt.ylabel('Frequency (Hz)')
    #     plt.title(f'Time-Frequency Representation - Epoch {i+1} (Gamma Band)')
    #     plt.show()



    # output_dir = 'Data/gamma_band_time_frequency_epochs'
    # os.makedirs(output_dir, exist_ok=True)

    # # Loop through each epoch in the list
    # for i, epoch_power_data in enumerate(all_epoch_powers):
    #     # epoch_power_data shape should be [n_channels, n_frequencies, n_time_points]
    #     for ch_idx, ch_name in enumerate(epochs.info['ch_names']):
    #         # Extract the power data for the specific channel
    #         channel_power_data = epoch_power_data[ch_idx]
            
    #         # Define filename for saving
    #         filename = f'epoch_{i+1}_channel_{ch_name}.npy'
            
    #         # Save the channel power data as a .npy file
    #         np.save(os.path.join(output_dir, filename), channel_power_data)

    # print(f"Time-frequency representations saved in {output_dir}")