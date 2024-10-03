import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Input layer to hidden layer
        self.fc2 = nn.Linear(128, 10)    # Hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))       # Activation function for hidden layer
        x = self.fc2(x)                    # Output layer
        return x

# Define the client class
class Client:
    def __init__(self, model, optimizer, criterion, data_loader):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.data_loader = data_loader

    def train(self, epochs=1):
        self.model.train()
        total_loss = 0.0
        for epoch in range(epochs):
            for data, target in self.data_loader:
                self.optimizer.zero_grad()
                output = self.model(data.view(data.size(0), -1))  # Flatten the data
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()  # Accumulate the loss
        return total_loss / len(self.data_loader)  # Return average loss for the epoch

    def get_weights(self):
        return {name: param.data.clone() for name, param in self.model.named_parameters()}

# Define the server class
class Server:
    def __init__(self, model, threshold):
        self.global_model = model
        self.threshold = threshold

    def aggregate_weights(self, client_weights):
        new_weights = {}
        for key in client_weights[0].keys():
            new_weights[key] = torch.mean(torch.stack([client_weights[i][key] for i in range(len(client_weights))]), dim=0)
        return new_weights

    def check_convergence(self, old_weights, new_weights):
        total_diff = 0
        for key in old_weights.keys():
            total_diff += torch.sum(torch.abs(old_weights[key] - new_weights[key])).item()
        print(f"Total difference: {total_diff}")  # Debugging info
        return total_diff < self.threshold

# Main function to simulate federated learning
def main():
    # Initialize the model and other parameters
    global_model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    threshold = 0.01  # Define a convergence threshold
    num_clients = 3
    rounds = 10  # Maximum number of rounds

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Create DataLoaders for each client using subsets of the dataset
    client_data_loaders = []
    data_size = len(dataset)
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    
    for i in range(num_clients):
        client_indices = indices[i::num_clients]  # Create subsets for each client
        client_subset = Subset(dataset, client_indices)
        client_data_loader = DataLoader(client_subset, batch_size=32, shuffle=True)
        
        optimizer = optim.SGD(global_model.parameters(), lr=0.01)
        client = Client(global_model, optimizer, criterion, client_data_loader)
        client_data_loaders.append(client)

    server = Server(global_model, threshold)

    # Training loop
    for round in range(rounds):
        print(f"\nRound {round + 1}")
        client_weights = []
        old_weights = {name: param.data.clone() for name, param in global_model.named_parameters()}  # Clone current weights

        for client in client_data_loaders:
            avg_loss = client.train(epochs=1)  # Train each client for 1 epoch
            print(f"Client average training loss: {avg_loss:.4f}")  # Print average loss for the client
            updated_weights = client.get_weights()
            client_weights.append(updated_weights)

        new_weights = server.aggregate_weights(client_weights)

        # Update global model
        for name, param in global_model.named_parameters():
            param.data = new_weights[name]

        # Check for convergence
        if server.check_convergence(old_weights, new_weights):
            print("Convergence reached. Stopping training.")
            break
        else:
            print("Updating global model parameters...")

if __name__ == "__main__":
    main()
