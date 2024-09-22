import torch
from torchvision import models, transforms
from PIL import Image

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(512, 10)  # Adjust for CIFAR-10 classes
model.load_state_dict(torch.load('C:\\Users\\joshi\\Desktop\\Pytorch\\cnn_cifar10_resnet.pth', weights_only=True))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 32x32 for CIFAR-10
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalization
])

# Load and preprocess the image
img_path = 'C:\\Users\\joshi\\OneDrive\\Desktop\\pexels-photo-1108099.jpeg'  # Replace with your image path
image = Image.open(img_path)
image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

# Perform inference
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

print(f'Predicted class: {predicted.item()}')
