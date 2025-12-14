import torch
import torch.nn as nn
import torch.nn.functional as F

class MoviePosterNet(nn.Module):
    def __init__(self, num_genres):
        super(MoviePosterNet, self).__init__()
        # 3 canaux d'entrée (RGB) au lieu de 1 (niveaux de gris)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Pour 224x224 -> après 3 poolings: 224 -> 112 -> 56 -> 28
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_genres)  # Nombre de genres dynamique

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def get_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        return x

if __name__ == '__main__':
    # Test du modèle
    x = torch.rand(16, 3, 224, 224)  # Batch de 16 images 224x224 RGB
    net = MoviePosterNet(num_genres=10)
    y = net(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (16, 10)
    print("Modèle testé avec succès!")