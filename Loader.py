import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



# Define the custom dataset class
class NpyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Directory with all the `.npy` files.
            transform (callable, optional): Optional transform to apply to each sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.file_names = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
       
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # Load .npy file
        file_path = os.path.join(self.data_dir, self.file_names[idx])
        data = np.load(file_path)  # Assuming each file contains a 2D array (n_freqs, n_times)
        
        # Add a channel dimension and convert to a PyTorch tensor
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Shape: (1, n_freqs, n_times)
        
        # Apply any transforms, if provided
        if self.transform:
            data = self.transform(data)
            
        return data #TODO: 

# Directory where .npy files are stored
data_dir = 'output_arrays'  # Replace with the actual path to your .npy files

# Create dataset and data loader
dataset = NpyDataset(data_dir)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)  # Adjust batch_size as needed

# Example: Loop through the DataLoader and feed data to the VAE
vae = VariationalAutoencoder(input_shape=(n_freqs, n_times), latent_dim=16)  # Replace with actual input shape
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

n_epochs = 10
for epoch in range(n_epochs):
    for batch in data_loader:
        # Forward pass
        reconstructed, mu, logvar = vae(batch)
        
        # Compute loss
        loss = vae_loss(reconstructed, batch, mu, logvar)
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")



if __name__=="__main__":
    
