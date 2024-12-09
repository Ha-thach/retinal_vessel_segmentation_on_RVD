import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import timm


# Custom dataset class for fractal dimension data
class FractalDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Check if the file exists
        if not os.path.exists(self.image_paths[idx]):
            print(f"File not found: {self.image_paths[idx]}")
            return None, None, None  # Return None if the file is missing

        # Open image and convert to grayscale
        image = Image.open(self.image_paths[idx]).convert("L")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label, self.image_paths[idx]  # Return the image path as well

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ViT
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Duplicate the single channel into three channels
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the grayscale levels
])

# Paths to CSV files and folders
current_dir = os.path.dirname(__file__)  # Get the script's directory
train_csv_path = os.path.join(current_dir, 'donnees_train.csv')
val_csv_path = os.path.join(current_dir, 'donnees_val.csv')
train_folder = "/data/nhthach/project/DATA/Retinal_Fractal/Original/train/mask"
val_folder = "/data/nhthach/project/DATA/Retinal_Fractal/Original/test/mask"

# Load training data
train_data = pd.read_csv(train_csv_path, sep=';')
train_data = train_data.dropna(subset=['image_name', 'fractal_dimension'])
train_image_paths = [os.path.join(train_folder, img_name) for img_name in train_data['image_name']]
train_labels = train_data['fractal_dimension'].tolist()

# Load validation data
val_data = pd.read_csv(val_csv_path, sep=';')
val_data = val_data.dropna(subset=['image_name', 'fractal_dimension'])
val_image_paths = [os.path.join(val_folder, img_name) for img_name in val_data['image_name']]
val_labels = val_data['fractal_dimension'].tolist()

# Create training and validation datasets
train_dataset = FractalDataset(train_image_paths, train_labels, transform=transform)
val_dataset = FractalDataset(val_image_paths, val_labels, transform=transform)

# Define a fixed batch size and learning rate
batch_size = 32# Set batch size
learning_rate = 2.715236059720235e-05  # The best learning rate after a lot of tests

# Define the regression model using ViT
class ViTRegressor(nn.Module):
    def __init__(self):
        super(ViTRegressor, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Linear(self.vit.head.in_features, 1)

    def forward(self, x):
        return self.vit(x)

# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=20):
    # path = "/data/nhthach/project/Seg/results/Fractal/SAMUNETD"
    # results_csv_path = os.path.join(path, 'evaluation_results.csv')
    #
    # # Write headers to the CSV file
    # with open(results_csv_path, 'w', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(['Epoch', 'MAE', 'MSE', 'R²', 'RMSE'])

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            if images is None:  # Skip the batch if the image is missing
                continue
            optimizer.zero_grad()
            outputs = model(images)
            labels = labels.clone().detach().float().view(-1, 1)  # Adjust label shape
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Calculate and display metrics for epochs between 3 and 20
        if epoch + 1 >= 3:
            model.eval()
            val_loss = 0.0
            all_labels = []
            all_predictions = []
            with torch.no_grad():
                for images, labels, _ in val_loader:
                    if images is None:  # Skip the batch if the image is missing
                        continue
                    outputs = model(images)
                    labels = labels.clone().detach().float().view(-1, 1)  # Adjust label shape
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    all_labels.extend(labels.numpy())
                    all_predictions.extend(outputs.squeeze().numpy())

            val_loss = val_loss / len(val_loader.dataset)

            # Calculate evaluation metrics
            mae = mean_absolute_error(all_labels, all_predictions)
            mse = mean_squared_error(all_labels, all_predictions)
            r2 = r2_score(all_labels, all_predictions)
            rmse = mean_squared_error(all_labels, all_predictions, squared=False)

            print(f'Validation Loss (Epoch {epoch + 1}): {val_loss:.4f}')
            print(f'Mean Absolute Error (MAE) (Epoch {epoch + 1}): {mae:.4f}')
            print(f'Mean Squared Error (MSE) (Epoch {epoch + 1}): {mse:.4f}')
            print(f'R-squared (R²) Score (Epoch {epoch + 1}): {r2:.4f}')
            print(f'Root Mean Squared Error (RMSE) (Epoch {epoch + 1}): {rmse:.4f}')

            # Save results to CSV after each epoch
            # with open(results_csv_path, 'a', newline='') as csvfile:
            #     csvwriter = csv.writer(csvfile)
            #     csvwriter.writerow([epoch + 1, mae, mse, r2, rmse])

# Create the model, loss function, and optimizer
for i in range(4):
    model = ViTRegressor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Adjust DataLoader batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=20)

    # Predict fractal dimensions for validation set images
    print("\nPredictions for validation set:")
    with torch.no_grad():
        for images, labels, image_paths in val_loader:
            if images is None:  # Skip if the image is missing
                continue
            outputs = model(images)
            predictions = outputs.squeeze().tolist()
            if isinstance(predictions, float):
                predictions = [predictions]
            for i, prediction in enumerate(predictions):
                print(f"Image: {image_paths[i]}, True: {labels[i]}, Predicted: {prediction}")
