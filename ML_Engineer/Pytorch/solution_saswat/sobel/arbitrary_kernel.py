import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10  
import matplotlib.pyplot as plt

class ArbitraryNN(nn.Module):   #the arbitrary kernel NN model
    def __init__(self):
        super(ArbitraryNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  
            nn.ReLU(),  # ReLU for non- linearitty
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  
            nn.ReLU(),  
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(32 * 32 * 32, 1024),  
            nn.ReLU(), 
            nn.Linear(1024, 32 * 32), 
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.view(-1, 1, 32, 32)  
        return x

# Prepare the dataset
transform = transforms.Compose([
    transforms.Grayscale(),  # Convertingg to grayscale
    transforms.ToTensor(),  # Converting to tensor
])

dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model, loss function, and optimizer
model = ArbitraryNN()
criterion = nn.MSELoss()  # Mean squared error loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

for epoch in range(4):  
    for images, _ in data_loader:
        images = images[:, 0:1, :, :] 

        outputs = model(images)

     
        Arbitrary_operator = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        Arbitrary_operator.weight.data = torch.tensor([[[[-0.7, 0.343, -22], [-0.34, 0.9, 2], [0, 2, 1]]]]).float()
        Arbitrary_images = Arbitrary_operator(images)

      
        loss = criterion(outputs, Arbitrary_images)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


model.eval()
with torch.no_grad():
    for i, (images, _) in enumerate(data_loader):
        if i == 5: 
            images = images[:, 0:1, :, :]  
            Arbitrary_images = Arbitrary_operator(images)
            predicted_images = model(images)

            plt.figure(figsize=(10, 5))
            for j in range(5):  # Display first 5 images
                plt.subplot(2, 5, j + 1)
                plt.imshow(Arbitrary_images[j].squeeze(0).cpu(), cmap='gray')
                plt.title('True Arbitrary')
                plt.axis('off')

                plt.subplot(2, 5, j + 6)
                plt.imshow(predicted_images[j].squeeze(0).cpu(), cmap='gray')
                plt.title('Predicted')
                plt.axis('off')

            plt.savefig('arbitrary_kernel_comparison.png')  
            plt.show()
            break
