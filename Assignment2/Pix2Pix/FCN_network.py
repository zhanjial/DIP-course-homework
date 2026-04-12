import torch.nn as nn
import torch

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        ### FILL: add more CONV Layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 64, 4, 2, 1), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True)    
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )


        # Decoder (Deconvolutional Layers)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, 4, 2, 1), 
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 8, 4, 2, 1),  
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.deconv_final = nn.Sequential(
            nn.ConvTranspose2d(8 + 8, 3, 4, 2, 1),      
            nn.Tanh() 
        )


        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function

    def forward(self, x):
        # Encoder forward pass
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
    
    # Decoder
        # Decoder forward pass
        d1 = self.deconv1(x4)
        
        d1_cat = torch.cat([d1, x3], dim=1) 
        d2 = self.deconv2(d1_cat)
        
        d2_cat = torch.cat([d2, x2], dim=1) 
        d3 = self.deconv3(d2_cat)
        
        d3_cat = torch.cat([d3, x1], dim=1) 
        
        ### FILL: encoder-decoder forward pass

        output = self.deconv_final(d3_cat)
        
        return output
    