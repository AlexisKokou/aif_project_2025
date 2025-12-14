import argparse
import os
from statistics import mean
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import MoviePosterNet

# Configuration du device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

def train(net, optimizer, loader, epochs=10, writer=None):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}')
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'Epoch {epoch+1}/{epochs} - loss: {mean(running_loss):.4f}')
        if writer is not None:
            writer.add_scalar('training loss', mean(running_loss), epoch)
    return mean(running_loss)

def test(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    return test_corrects / total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='MoviePoster', help='experiment name')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--data_path', type=str, default='./content/sorted_movie_posters_paligema', help='path to dataset')
    
    args = parser.parse_args()
    
    # Transformations pour les images couleur
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Charger le dataset avec ImageFolder
    print(f"Loading dataset from: {args.data_path}")
    full_dataset = torchvision.datasets.ImageFolder(
        root=args.data_path,
        transform=transform
    )
    
    # Afficher les infos du dataset
    print(f"Dataset loaded successfully!")
    print(f"Total images: {len(full_dataset)}")
    print(f"Genres: {full_dataset.classes}")
    print(f"Class mapping: {full_dataset.class_to_idx}")
    
    # Split train/test (80/20)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    print(f"\nDataset split:")
    print(f"   Training: {len(train_dataset)} images")
    print(f"   Testing: {len(test_dataset)} images")
    
    # Créer les dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Nombre de genres
    num_genres = len(full_dataset.classes)
    
    # Créer le modèle
    net = MoviePosterNet(num_genres).to(device)
    print(f"\nModel created with {num_genres} output classes")
    
    # TensorBoard writer
    writer = SummaryWriter(f'runs/{args.exp_name}')
    
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    # Entraînement
    train(net, optimizer, trainloader, args.epochs, writer)
    
    # Test
    test_acc = test(net, testloader)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Sauvegarder le modèle
    if not os.path.exists('./weights'):
        os.makedirs('./weights')
    
    # Sauvegarde complète avec métadonnées
    checkpoint = {
        'model_state_dict': net.state_dict(),
        'classes': full_dataset.classes,
        'class_to_idx': full_dataset.class_to_idx,
        'num_genres': num_genres,
        'test_accuracy': test_acc
    }
    
    model_path = f'./weights/movie_poster_net.pth'
    torch.save(checkpoint, model_path)
    print(f"\nModel saved to: {model_path}")
    print(f"   Classes saved: {full_dataset.classes}")
    
    # TensorBoard - hyperparameters
    writer.add_hparams(
        {'lr': args.lr, 'bsize': args.batch_size, 'epochs': args.epochs},
        {'hparam/accuracy': test_acc}
    )
    
    writer.close()
    print("\nTraining completed successfully!")