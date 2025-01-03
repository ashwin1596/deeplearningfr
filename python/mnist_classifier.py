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
config.set_batch_size(32)
config.set_num_epochs(50)

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
        labels = np.eye(num_classes)[labels].astype(np.float32)

        # Convert to Tensor
        input_tensor = dl.Tensor(images, False)
        target_tensor = dl.Tensor(labels, False)

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


def evaluate_model(model, test_loader, builder, num_classes=10):
        """
        Evaluate the model on the test dataset with gradient computation disabled.
    
        Args:
            model: The trained neural network model
            test_loader: DataLoader containing test data
            builder: GraphBuilder instance
            num_classes: Number of output classes (default: 10 for MNIST)
    
        Returns:
            tuple: (accuracy, test_loss)
        """
        correct = 0
        total = 0
        total_loss = 0.0
    
        # Create loss function for evaluation
        loss_fn = dl.CrossEntropyLoss(builder)
    
        # Disable gradient computation during evaluation
        with dl.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                # Prepare input data
                images = images.view(images.size(0), -1).numpy().astype(np.float32)
                original_labels = labels.numpy().astype(int)
            
                # One-hot encode the labels
                labels_one_hot = np.eye(num_classes)[original_labels].astype(np.float32)
            
                # Convert to Tensors
                input_tensor = dl.Tensor(images, True)
                target_tensor = dl.Tensor(labels_one_hot, True)
            
                # Create graph nodes
                input_node = builder.createVariable("input", input_tensor.transpose())
                target_node = builder.createVariable("target", target_tensor.transpose())
            
                # Forward pass
                outputs = model.forward(input_node)
            
                # Compute loss
                loss = loss_fn.forward(outputs, target_node)
                total_loss += loss.value().get_data()[0]
            
                # Get predictions
                predictions = outputs.value().get_data()
                predicted_labels = np.argmax(predictions, axis=0)
            
                # Update accuracy metrics
                total += labels.size(0)
                correct += np.sum(predicted_labels == original_labels)
            
                if batch_idx % 10 == 0:
                    print(f'Testing batch {batch_idx}/{len(test_loader)}, ' 
                          f'Current accuracy: {100 * correct/total:.2f}%')
    
        # Calculate final metrics
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(test_loader)
    
        print(f'\nTest Results:')
        print(f'Average Loss: {avg_loss:.4f}')
        print(f'Accuracy: {accuracy:.2f}%')
    
        return accuracy, avg_loss

evaluate_model(model, test_loader, builder, 10)