import deeplearning as dl
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.ToTensor()  # Convert PIL image to tensor
    # ,transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
])  

config = dl.Config.get_instance()

config.set_device_type('GPU')
config.set_cuda_devices('0')
config.set_batch_size(1024)
config.set_num_epochs(10)

batch_size = config.get_batch_size()
num_epochs = config.get_num_epochs()
num_classes = 10

# Download and transform the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

builder = dl.GraphBuilder()
model = dl.Sequential(builder)
model.add_layer(dl.Linear(28 * 28, 256, builder, layer_num=1))  # Input: 784, Output: 128
model.add_layer(dl.ReLU(builder, layer_num=2))
model.add_layer(dl.Linear(256, 128, builder, layer_num=3))  #
model.add_layer(dl.ReLU(builder, layer_num=4))
model.add_layer(dl.Linear(128, 10, builder, layer_num=5))  #
# model.add_layer(dl.Linear(28 * 28, 128, builder, layer_num=1))  # Input: 784, Output: 128
# model.add_layer(dl.ReLU(builder, layer_num=2))
# model.add_layer(dl.Linear(128, 10, builder, layer_num=3))  # Output: 10 (for 10 digits)

loss_fn = dl.CrossEntropyLoss(builder)  # Loss function
parameters = model.parameters()  # Get model parameters
optimizer = dl.SGD(parameters, learning_rate=0.001)  # Optimizer



for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        # Flatten the images to match the input dimension (batch_size, 28*28)
        images = images.view(images.size(0), -1).numpy().astype(np.float32)  # Convert to numpy array
        labels = labels.numpy().astype(int)  # Convert labels to numpy array

        # One-hot encode the labels
        labels = np.eye(num_classes)[labels]

        # Convert to Tensor
        input_tensor = dl.Tensor([images.shape[0], 28 * 28], images.flatten().tolist())
        target_tensor = dl.Tensor(labels.shape, labels.flatten().tolist())

        input_node = builder.createVariable("input", input_tensor.transpose())
        target_node = builder.createVariable("target", target_tensor.transpose())

        # Forward pass
        outputs = model.forward(input_node)

        # Compute loss
        loss = loss_fn.forward(outputs, target_node)

        # Backward pass
        builder.backward(loss)
        optimizer.step()

        # Accumulate loss
        total_loss += loss.value().get_data()[0]

        print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}, Loss: {loss.value().get_data()[0]}")
        	
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
