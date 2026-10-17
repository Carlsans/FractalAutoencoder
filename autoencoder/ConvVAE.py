import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=256, beta = .1):
        super(ConvVAE, self).__init__()
        self.beta = beta  # KL weight factor for beta-VAE
        # Encoder
        self.encoder_conv = nn.Sequential(
            self._conv_layer(1, 32, 4, 2, 1),  # Output: 32 x 256 x 256
            self._conv_layer(32, 64, 4, 2, 1),  # Output: 64 x 128 x 128
            self._conv_layer(64, 128, 4, 2, 1),  # Output: 128 x 64 x 64
            self._conv_layer(128, 256, 4, 2, 1),  # Output: 256 x 32 x 32
            self._conv_layer(256, 512, 4, 2, 1),  # Output: 512 x 16 x 16
            self._conv_layer(512, 1024, 4, 2, 1),  # Output: 1024 x 8 x 8
            nn.Flatten()
        )

        # Linear layers for mean and log variance
        self.fc_mu = nn.Linear(1024 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 8 * 8, latent_dim)

        # Initialize the linear layers
        nn.init.xavier_normal_(self.fc_mu.weight, gain=0.01)
        nn.init.xavier_normal_(self.fc_logvar.weight, gain=0.01)
        nn.init.constant_(self.fc_mu.bias, 0)
        nn.init.constant_(self.fc_logvar.bias, 0)

        # Decoder
        self.decoder_linear = nn.Linear(latent_dim, 1024 * 8 * 8)
        self.decoder_conv = nn.Sequential(
            self._conv_transpose_layer(1024, 512, 4, 2, 1),  # Output: 512 x 16 x 16
            self._conv_transpose_layer(512, 256, 4, 2, 1),  # Output: 256 x 32 x 32
            self._conv_transpose_layer(256, 128, 4, 2, 1),  # Output: 128 x 64 x 64
            self._conv_transpose_layer(128, 64, 4, 2, 1),  # Output: 64 x 128 x 128
            self._conv_transpose_layer(64, 32, 4, 2, 1),  # Output: 32 x 256 x 256
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # Output: 1 x 512 x 512
            nn.Sigmoid()
        )

        # Initialize the decoder linear layer
        nn.init.xavier_normal_(self.decoder_linear.weight, gain=0.01)
        nn.init.constant_(self.decoder_linear.bias, 0)


    def _conv_layer(self, in_channels, out_channels, kernel_size, stride, padding):
            """Helper function to create a Conv2d layer with initialization and batch norm"""
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
            # Initialize the conv layer
            nn.init.kaiming_normal_(layer[0].weight, mode='fan_out', nonlinearity='leaky_relu')
            return layer

    def _conv_transpose_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        """Helper function to create a ConvTranspose2d layer with initialization and batch norm"""
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Initialize the conv transpose layer
        nn.init.kaiming_normal_(layer[0].weight, mode='fan_out', nonlinearity='leaky_relu')
        return layer

    def encode(self, x):
        x = self.encoder_conv(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        z = self.decoder_linear(z)
        z = z.view(-1, 1024, 8, 8)
        return self.decoder_conv(z)

    def forward(self, x):
        #print(x)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def loss_function(self, reconstructed, x, mu, logvar, kl_weight_annealing=1.0):
        """
        Loss function with KL weight annealing and scaled reconstruction loss
        """
        # Ensure reconstructed values are in valid range
        reconstructed = torch.clamp(reconstructed, 1e-6, 1.0 - 1e-6)

        # Check for NaN values
        if torch.isnan(reconstructed).any() or torch.isnan(x).any():
            raise ValueError("NaN values detected in input or reconstruction")

        # Reconstruction loss (binary cross entropy)
        recon_loss = F.binary_cross_entropy(reconstructed, x, reduction='sum')

        # Scale reconstruction loss by image dimensions to balance with KL loss
        recon_loss = recon_loss / x.size(0)  # Divide by batch size

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Divide by batch size

        # Apply beta factor and KL annealing
        kl_weight = self.beta * kl_weight_annealing

        # Total loss
        total_loss = recon_loss + kl_weight * kl_loss

        return total_loss, recon_loss, kl_loss

