import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models

class UECFOOD256Dataset(Dataset):
    def __init__(self, data_folder, category_file, transform=None):
        self.data_folder = data_folder
        self.category_file = category_file
        self.image_files = []
        self.labels = []
        with open(category_file, 'r') as f:
            next(f)  # Skip the header
            for line in f:
                label_id, label_name = line.strip().split('\t')
                label_id = int(label_id)
                label_folder = os.path.join(data_folder, str(label_id))
                for file in os.listdir(label_folder):
                    if file.endswith('.jpg'):
                        self.image_files.append(os.path.join(label_folder, file))
                        self.labels.append(label_id - 1)  # Adjust labels to start from 0
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Define the data transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_folder = './UECFOOD256'
category_file = './UECFOOD256./category.txt'

# Load the dataset
dataset = UECFOOD256Dataset(data_folder, category_file, transform=data_transform)

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print('train_loader: ',train_loader)
print('test_loader :',test_loader)

# Load the pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Modify the last fully connected layer to match the number of classes (256 in this case)
num_classes = 256
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    print('inside epoch')
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total}%')
