import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim=16):
        super(Encoder, self).__init__()
        n_freqs, n_times = input_shape
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # Output: 32 x n_freqs/2 x n_times/2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Output: 64 x n_freqs/4 x n_times/4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # Output: 128 x n_freqs/8 x n_times/8
        
        # Calculate the size of the feature map after convolutions
        conv_output_size = (n_freqs // 8) * (n_times // 8) * 128
        
        # Fully connected layers for mean and log-variance
        self.fc_mu = nn.Linear(conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(conv_output_size, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output
        x = torch.flatten(x, start_dim=1)
        
        # Get mean and log-variance
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, input_shape, latent_dim=16):
        super(Decoder, self).__init__()
        n_freqs, n_times = input_shape

        # Calculate size after convolutions for reshaping
        conv_output_size = (n_freqs // 8) * (n_times // 8) * 128
        
        # Fully connected layer for reconstructing feature map shape
        self.fc = nn.Linear(latent_dim, conv_output_size)
        
        # Deconvolution layers
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        # Decode fully connected to a feature map
        x = self.fc(z)
        x = x.view(-1, 128, n_freqs // 8, n_times // 8)
        
        # Apply deconvolutions
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  # Output in [0, 1] range
        
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_shape, latent_dim=16):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(input_shape, latent_dim)
        self.decoder = Decoder(input_shape, latent_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)  # Random noise
        return mu + eps * std
    
    def forward(self, x):
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decoder(z)
        
        return reconstructed, mu, logvar

# Define loss function
def vae_loss(reconstructed, original, mu, logvar):
    # Reconstruction loss (Binary Cross Entropy)
    recon_loss = F.binary_cross_entropy(reconstructed, original, reduction='sum')
    
    # KL Divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss