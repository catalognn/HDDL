from itertools import product
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import itertools
print("Hello World")
print("aa")
import os
import random
import requests
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import Counter
import torch.nn.functional as F
import csv


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Après chaque époque
torch.cuda.empty_cache()


data_dir = ['bottle', 'hazelnut', 'toothbrush', 'engine_wiring', 'capsule']
im_train = []
im_test = [[] for _ in range(len(data_dir))]
i=0
for data in data_dir:
    print("######")
    print(data)
    # Définir les transformations des images (normalisation, redimensionnement, etc.)
    transform = transforms.Compose([
        transforms.Resize((400, 400)), #Redimensionner les images en se basant sur la taille d'images la plus petite du dataset.
        transforms.ToTensor(),  # Convertir les images en tenseurs
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation standard pour les modèles pré-entraînés
    ])
    
    # Charger le dataset en spécifiant le dossier train ou test
    train_dataset = datasets.ImageFolder(root=f"{data}/train", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data}/test", transform=transform)

    # Créer des variables distinctes pour les DataLoaders
    exec(f"{data}_train_dataset = train_dataset")
    exec(f"{data}_test_dataset = test_dataset")
    
    # Créer les DataLoaders
    batch_size = 16
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Créer des variables distinctes pour les DataLoaders
    exec(f"{data}_train_loader = train_loader")
    exec(f"{data}_test_loader = test_loader")

    
    # Vérifier le nombre d'images par dataset
    print(f"Nombre d'images pour le train : {len(train_dataset)}")
    im_train.append(len(train_dataset))
    print(f"Nombre d'images pour le test : {len(test_dataset)}")
    
    # Classes
    print(f"Classes détectées dans train : {train_dataset.classes}")
    print(f"Classes détectées dans test : {test_dataset.classes}")

    # Récupérer les indices des classes pour chaque image
    class_counts = Counter(test_dataset.targets)
    
    # Obtenir les noms des classes
    class_names = test_dataset.classes
    
    # Afficher les résultats
    for class_idx, count in class_counts.items():
        print(f"Classe '{class_names[class_idx]}': {count} images")
        im_test[i].append(count)
    i+=1

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(),
            nn.Conv2d(256, latent_dim, kernel_size=4, stride=2, padding=1),  # 4x4 -> 2x2
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1),  # 2x2 -> 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


class ColorizationModel(nn.Module):
    def __init__(self, latent_dim=128):
        super(ColorizationModel, self).__init__()
        self.encoder = Encoder(latent_dim=latent_dim, in_channels=1)  # Input grayscale
        self.decoder = Decoder(latent_dim=latent_dim, out_channels=3)  # Predict RGB

    def forward(self, x):
        grayscale_x = transforms.Grayscale()(x)  # Convert RGB to Grayscale
        z = self.encoder(grayscale_x)
        return self.decoder(z), grayscale_x

class InpaintingModel(nn.Module):
    def __init__(self, latent_dim=128, mask_size=8):
        super(InpaintingModel, self).__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.mask_size = mask_size
    
    #forward method
    def forward(self, x):
        x_masked = self.apply_mask(x)
        z = self.encoder(x_masked)
        return self.decoder(z), x_masked

    #mask method
    def apply_mask(self, x):
        masked_x = x.clone()

        for i in range(masked_x.size(0)):
            ul_x = np.random.randint(0, x.size(2) - self.mask_size + 1) #Randomly sample the x coordinate of the upper left corner
            ul_y = np.random.randint(0, x.size(3) - self.mask_size + 1) #Randomly sample the y coordinate of the upper left corner
            masked_x[i, :, ul_x:ul_x+self.mask_size, ul_y:ul_y+self.mask_size] = 0

        return masked_x

class MaskedAutoencoderModel(nn.Module):
    def __init__(self, latent_dim=128, mask_ratio=1/16):
        super(MaskedAutoencoderModel, self).__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.mask_ratio = mask_ratio

    def forward(self, x):
        x_masked = self.apply_mask(x)
        z = self.encoder(x_masked)
        return self.decoder(z), x_masked
    
    def apply_mask(self, x):
        x_masked = x.clone()
        mask = torch.rand_like(x[:, 0, :, :]) < self.mask_ratio
        mask = mask.unsqueeze(1).repeat(1, x.size(1), 1, 1)
        x_masked[mask] = 0
        return x_masked


def train_ssl_model(model, 
                    train_loader, 
                    test_loader, 
                    criterion,
                    optimizer,
                    device='cuda',
                    epochs=5,
                    model_name="Model",
                    dataset_name="Dataset",
                    param_dict=None,
                    save_path="results.csv"):
    # Initialiser les listes pour stocker les pertes
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.to(device)
        model.train()
        total_train_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            output, _ = model(images)
            loss = criterion(output, images)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)  # Stocker la perte d'entraînement

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                output, _ = model(images)
                val_loss = criterion(output, images)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)  # Stocker la perte de validation

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

        # Sauvegarde dans un fichier CSV toutes les 5 époques
        if (epoch + 1) % 5 == 0:
            with open(save_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    model_name, dataset_name, epoch + 1, 
                    param_dict, avg_train_loss, avg_val_loss
                ])
        # Après chaque époque
        torch.cuda.empty_cache()

    return model.encoder, avg_val_loss



# Adapter le dictionnaire de paramètres au modèle courant
def get_model_specific_param_grid(model_name, param_grid):
    param_grid = param_grid.copy()  # Créer une copie pour éviter d'affecter l'original
    if model_name != "MaskedAutoencoder":
        param_grid.pop('mask_ratio', None)  # Retirer 'mask_ratio' si non pertinent
    if model_name != "Inpainting":
        param_grid.pop('mask_size', None)  # Retirer 'mask_size' si non pertinent
    return param_grid


from itertools import product

# Adapter le dictionnaire de paramètres au modèle courant
def get_model_specific_param_grid(model_name, param_grid):
    param_grid = param_grid.copy()  # Créer une copie pour éviter d'affecter l'original
    if model_name != "MaskedAutoencoder":
        param_grid.pop('mask_ratio', None)  # Retirer 'mask_ratio' si non pertinent
    if model_name != "Inpainting":
        param_grid.pop('mask_size', None)  # Retirer 'mask_size' si non pertinent
    return param_grid


def grid_search_ssl(models, datasets, param_grid, criterion, device='cuda', epochs=5, save_path="results.csv"):
    """
    Grid Search for Self-Supervised Learning models across multiple datasets.

    Args:
        models (dict): Dictionary of models with their names as keys.
        datasets (dict): Dictionary with dataset names as keys and (train_loader, test_loader) tuples as values.
        param_grid (dict): Hyperparameter grid (keys are hyperparameters, values are lists of possible values).
        criterion: Loss function.
        device (str): Device to use ('cuda' or 'cpu').
        epochs (int): Number of epochs to train.
        save_path (str): File path for saving results.

    Returns:
        results (dict): Best parameters and corresponding validation loss for each model and dataset.
    """
    results = {}

    # Générer l'en-tête du fichier CSV
    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Dataset", "Epoch", "Parameters", "Train Loss", "Validation Loss"])

    # Generate all combinations of hyperparameters
    param_combinations = list(product(*param_grid.values()))

    for model_name, model_class in models.items():
        print(f"==== Grid Search for Model: {model_name} ====")
        results[model_name] = {}

        for dataset_name, (train_loader, test_loader) in datasets.items():
            print(f"  Dataset: {dataset_name}")
            best_loss = float('inf')
            best_params = None

            # Obtenir le param_grid spécifique au modèle
            param_grid = get_model_specific_param_grid(model_name, param_grid)

            # Générer toutes les combinaisons de paramètres pertinentes
            param_combinations = list(itertools.product(*param_grid.values()))

            for params in param_combinations:                
                # Create the model with current hyperparameters
                param_dict = dict(zip(param_grid.keys(), params))
                model_params = {k: v for k, v in param_dict.items() if k not in ['lr']}
                if model_name != "MaskedAutoencoder" :
                    model_params = {k: v for k, v in model_params.items() if k not in ['mask_ratio']}
                if model_name != "Inpainting" :
                    model_params = {k: v for k, v in model_params.items() if k not in ['mask_size']}
                # Afficher la combinaison de paramètres testée
                print(f"Testing parameters: {param_dict}")
                model = model_class(**model_params).to(device)

                # Define optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=param_dict['lr'])

                # Train the model
                trained_encoder, val_loss = train_ssl_model(
                    model,
                    train_loader,
                    test_loader,
                    criterion,
                    optimizer,
                    device=device,
                    epochs=epochs,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    param_dict=param_dict,
                    save_path=save_path
                )

                # Check if current configuration is the best
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = param_dict

            # Store best parameters and loss
            results[model_name][dataset_name] = {
                'best_params': best_params,
                'best_val_loss': best_loss
            }

            # Ajouter les meilleurs résultats au CSV
            with open(save_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    model_name, dataset_name, "Best", best_params, "-", best_loss
                ])

            print(f"    Best Params: {best_params}, Best Val Loss: {best_loss:.4f}")

    return results

models = {
    'MaskedAutoencoder': lambda **params: MaskedAutoencoderModel(**params),
    'Colorization': lambda **params: ColorizationModel(**params),
    'Inpainting': lambda **params: InpaintingModel(**params),
}


datasets = {
    'bottle': (bottle_train_loader, bottle_test_loader),
    'hazelnut': (hazelnut_train_loader, hazelnut_test_loader),
    'toothbrush': (toothbrush_train_loader, toothbrush_test_loader),
    'engine_wiring': (engine_wiring_train_loader, engine_wiring_test_loader),
    'capsule': (capsule_train_loader, capsule_test_loader),
}


param_grid = {
    'lr': [0.01, 0.005, 0.001, 0.0001],
    'latent_dim': [64, 128, 256],
    'mask_size': [16, 32, 64],  # For InpaintingModel only
    'mask_ratio': [1/16, 1/8, 1/4],  # For MaskedAutoencoderModel only
}

criterion = nn.MSELoss()
results = grid_search_ssl(models, datasets, param_grid, criterion, device='cpu', epochs=40)
print(results)