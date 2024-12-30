import deeplearning as dl

config = dl.Config.get_instance()

config.set_device_type('GPU')
config.set_cuda_devices('0')
config.set_batch_size(32)
config.set_num_epochs(10)


# Create components
builder = dl.GraphBuilder()
model = dl.Sequential(builder)

# Add layers
linear_layer1 = dl.Linear(784, 128, builder, 1)

model.add_layer(linear_layer1)
model.add_layer(dl.ReLU(builder, 1))
model.add_layer(dl.Linear(128, 10, builder, 2))

# Create dataset and dataloader
train_dataset = dl.MNISTDataset("data/mnist")
train_loader = dl.DataLoader(train_dataset, 32, True)

# Create optimizer and loss
optimizer = dl.SGD(model.parameters(), 0.01)
loss_fn = dl.CrossEntropyLoss(builder)

total_epochs = config.get_num_epochs()
# Training loop
for epoch in range(total_epochs):
    print(f"Epoch {epoch + 1} starting...")
    
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        print(type(batch.inputs))

        # Create input and target nodes
        input_node = builder.createVariable("input", batch.inputs.transpose())
        target_node = builder.createVariable("target", batch.targets.transpose())
        
        # Forward pass
        logits = model.forward(input_node)
        loss = loss_fn.forward(logits, target_node)
        
        # Backward pass
        builder.backward(loss)
        
        # Update parameters
        optimizer.step()
        
        print(f"Epoch: {epoch + 1}, Batch: {batch_idx + 1}/{train_loader.total_batches()}, Loss: {loss.value()[0]}")

# Save model
model.save_model("model-1.json")