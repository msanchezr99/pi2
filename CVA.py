import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_shape, interm_dim=150, latent_dim=16,variational=False):
        super(Encoder, self).__init__()
        self.variational=variational
        n_freqs, n_times = input_shape

        # Capas convolucionales
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3,5),stride=2)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3,stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 2))##Comentar
        self.bn2= nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(8, 16,kernel_size=3,stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn2 = nn.BatchNorm2d(32)
        # Determinación de shape luego de convolución
        self.conv_c_out,self.conv_h_out,self.conv_w_out=self._get_conv_output(input_shape)
        self.conv_output_size = self.conv_c_out*self.conv_h_out*self.conv_w_out
        
        # Capas fully connected para espacio latente
        self.fc_1=nn.Linear(self.conv_output_size,interm_dim)
        self.fc_mu = nn.Linear(interm_dim, latent_dim) #De representación intermedia a latente
        if self.variational:
            self.fc_logvar = nn.Linear(self.interm_dim, latent_dim)
        print("Instanciado encoder")

    def _get_conv_output(self, input_shape):
        """Función auxiliar en cálculo de tamaño tras convoluciones
        Retorna (C_out,H_out,W_out): numero de canales, altura y ancho de salida."""
        with torch.no_grad():
            x = torch.zeros(1, 1, *input_shape)  # Create a dummy input with batch size 1
            x = self.conv1(x)
            #x = self.maxpool1(x)
            x = self.conv2(x)
            #x = self.maxpool2(x)
            x = self.conv3(x)
            x = self.maxpool1(x)
            #x = self.maxpool3(x)
        
            return x.shape[1],x.shape[2],x.shape[3]
            
    def forward(self, x):
    # print(f"Input dimensions: {x.shape}")
        x = F.relu(self.conv1(x))
    # print(f"Encoder conv1: {x.shape}")
        
        x = F.relu(self.conv2(x))
        #x = self.maxpool2(x)
    # print(f"Encoder capa 2: {x.shape}")
        x = F.relu(self.conv3(x))
        #x = self.maxpool3(x)
    # print(f"Encoder capa 3: {x.shape}")
        x = self.maxpool1(x)
    # print(f"Encoder maxpool: {x.shape}")
        # Flatten the output
        x = torch.flatten(x, start_dim=1)
    # print("Encoder Forward: Flatten alcanzado. x.hape: ",x.shape, "Predicted output size: ", self.conv_output_size)
        x =F.relu(self.fc_1(x))
    # print("Linear 1: ",x.shape)
        # Obtener parámetros mu, logvarianza
        mu = self.fc_mu(x)
    # print("Mu alcanzado",mu.shape)
        if self.variational:
            logvar = self.fc_logvar(x)
            return mu,logvar
        return mu

class Decoder(nn.Module):
    def __init__(self, input_shape, encoder_conv_output_shape,interm_dim=150,latent_dim=16):
        super(Decoder, self).__init__()
        self.n_freqs, self.n_times = input_shape
        # Shape luego de convoluciones en el encoder para hacer reshaping
        self.conv_output_shape = encoder_conv_output_shape
        print(self.conv_output_shape)
        encoder_conv_output_size=self.conv_output_shape[0]*self.conv_output_shape[1]*self.conv_output_shape[2]
        # Fully connected layer for reconstructing feature map shape
        self.fc=nn.Linear(latent_dim,interm_dim)
        self.fc_1 = nn.Linear(interm_dim, encoder_conv_output_size)#C*H*W
        #En el forward, se hace el view y se obtiene dimensión (C,H,W) de representación intermedia
        
        # Deconvolution layers
        self.deconv1 = nn.ConvTranspose2d(16, 8, kernel_size=3,stride=2,output_padding=1)
        self.upsample1 = nn.Upsample(scale_factor=(1, 2))
        self.deconv2 = nn.ConvTranspose2d(8, 8, kernel_size=3,stride=2,output_padding=(0,1))
        self.upsample2 = nn.Upsample(scale_factor=(2, 2))
        self.deconv3 = nn.ConvTranspose2d(8, 1, kernel_size=(3,5),stride=2,output_padding=1)
        print("Instanciado decoder")

    def forward(self, x):#x: entrada de espacio latente
        # Decode fully connected to a feature map
        x = F.relu(self.fc(x))
    # print(f"Decoder capa lineal 1: {x.shape}")
        x = F.relu(self.fc_1(x))
    # print(f"Decoder capa lineal 2: {x.shape}")
        
        # El 1 se refiere a a la dimensión 
        x = x.unflatten(1, self.conv_output_shape) # "desarma" la dimensión 1 en (C,H,W)
        print(f"Decoder unflattened: {x.shape}")
        
        x = self.upsample1(x)
    # print(f"Decoder upsample: {x.shape}")
        # Apply deconvolutions
        x = F.relu(self.deconv1(x))
        
    # print(f"Decoder capa 1: {x.shape}")
        
        x = F.relu(self.deconv2(x))
        
    # print(f"Decoder capa 2: {x.shape}")
        #x = self.upsample1(x)
        #print(f"Decoder upsample: {x.shape}")
        x = torch.sigmoid(self.deconv3(x))
        #x = self.upsample2(x)
    # print(f"Decoder capa 3: {x.shape}")

        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_shape, interm_dim=150, latent_dim=16,variational=False):
        super(VariationalAutoencoder, self).__init__()
        self.variational=variational
        self.encoder = Encoder(input_shape, interm_dim=interm_dim,latent_dim=latent_dim,variational=variational)
        encoder_conv_c_h_w_out=(self.encoder.conv_c_out,self.encoder.conv_h_out,self.encoder.conv_w_out)
        self.decoder = Decoder(input_shape,encoder_conv_c_h_w_out, interm_dim=interm_dim,latent_dim=latent_dim)
     
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)  # Random noise
        return mu + eps * std
    
    def forward(self, x):
        if self.variational:# Encode
            mu, logvar = self.encoder(x)
            # Reparameterize
            z = self.reparameterize(mu, logvar)
            # Decode
            reconstructed = self.decoder(z)
            return reconstructed, mu, logvar
        else: 
            z=self.encoder(x)
            reconstructer=self.decoder(z)
            return reconstructed
# Define loss function
def loss_funct(reconstructed, original, mu=None, logvar=None, kld_weight=0.1,variational=False):
    """
    Compute VAE loss with weighted KL divergence
    """
    recon_loss = F.binary_cross_entropy(reconstructed, original, reduction='sum')
    if variational:
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld_weight * kld_loss
    else:
        return recon_loss