import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoderwithClassifier(nn.Module):
    def __init__(self,classes, latent_dim=256):
        super(ConvAutoencoderwithClassifier, self).__init__()
        self.classcount = len(classes)
        self.classes = classes
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # Output: 32 x 256 x 256
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: 64 x 128 x 128
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: 128 x 64 x 64
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: 256 x 32 x 32
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Output: 512 x 16 x 16
            nn.ReLU(True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # Output: 1024 x 8 x 8
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(1024 * 8 * 8, latent_dim)  # Latent space
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024 * 8 * 8),
            nn.Unflatten(1, (1024, 8, 8)),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # Output: 512 x 16 x 16
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Output: 256 x 32 x 32
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: 128 x 64 x 64
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 64 x 128 x 128
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: 32 x 256 x 256
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # Output: 1 x 512 x 512
            nn.Sigmoid()  # To keep output pixel values between 0 and 1
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.ReLU(),
            nn.Linear(latent_dim*2, self.classcount)#,
            #nn.Tanh(),
        )
    def forward(self, x):
        # Encode input to latent space
        latent = self.encoder(x)
        logits = self.classifier(latent)
        # Decode from latent space to reconstructed image
        reconstructed = self.decoder(latent)
        return reconstructed, latent, logits


# # Instantiate the model
# latent_dim = 128  # Size of the latent space for potential classification
# model = ConvAutoencoder(latent_dim=latent_dim)
#
# # Example usage with a grayscale image batch of shape [batch_size, 1, 512, 512]
# x = torch.randn(8, 1, 512, 512)  # Example input batch
# reconstructed, latent = model(x)
# print("Reconstructed shape:", reconstructed.shape)  # Expected: [8, 1, 512, 512]
# print("Latent space shape:", latent.shape)  # Expected: [8, 128]
