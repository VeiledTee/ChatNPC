import torch
import torch.nn as nn
import torch.optim as optim
from BiLSTM import BiLSTMModel

# Assume you have your training data (X_train, y_train) ready

# Define hyperparameters
input_size = 10  # Example input size
hidden_size = 64  # Example hidden size
num_layers = 2  # Example number of layers
output_size = 1  # Example output size
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Create an instance of the BiLSTMModel
model = BiLSTMModel(input_size, hidden_size, num_layers, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Convert the training data into tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Create data loader for batching the training data
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Set the model in training mode
model.train()

# Training loop
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss for every few iterations
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Iteration [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# After training, you can save the model if needed
torch.save(model.state_dict(), 'bilstm_model.pth')

