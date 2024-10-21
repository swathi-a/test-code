import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from botorch.test_functions import SyntheticTestFunction
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


# Define a CNN model for classification
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # Assuming 10 classes for classification

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the problem inheriting from SyntheticTestFunction
class MPIImageClassificationProblem(SyntheticTestFunction):
    def __init__(self, train_loader, test_loader, num_epochs=10):
        super().__init__(noise_std=None)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.model = SimpleCNN()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def forward(self, params):
        try:
            lr = params[0].item()
            momentum = params[1].item()

            # Set optimizer with new learning rate and momentum
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
            return self.train_and_evaluate()
        except Exception as e:
            print(f"An error occurred during training: {str(e)}")
            return float('inf')  # Return a high error to avoid selecting this configuration

    def train_and_evaluate(self):
        try:
            # Train the model
            for epoch in range(self.num_epochs):
                self.model.train()
                for images, labels in self.train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

            # Evaluate on the test set
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in self.test_loader:
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total
            return -accuracy  # Return negative accuracy for optimization

        except Exception as e:
            print(f"An error occurred during evaluation: {str(e)}")
            return float('inf')  # Return a high error in case of failure


# Load CIFAR-10 dataset (as a stand-in for MPI dataset)
def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader


# Perform Bayesian optimization on the problem
def bayesian_optimization(problem, bounds, n_iter=20, n_initial_points=5):
    # Initial random points
    train_x = torch.rand(n_initial_points, 2) * (bounds[1] - bounds[0]) + bounds[0]
    train_y = torch.tensor([problem(x) for x in train_x]).unsqueeze(-1)
    regrets = []

    # Train GP model
    for i in range(n_iter):
        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        # Upper Confidence Bound acquisition function
        UCB = UpperConfidenceBound(model, beta=0.1)

        # Optimize acquisition function
        candidate, _ = optimize_acqf(UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20)
        candidate = candidate.detach()
        y_new = problem(candidate).unsqueeze(-1)

        # Update dataset
        train_x = torch.cat([train_x, candidate])
        train_y = torch.cat([train_y, y_new])

        # Calculate regret
        regret = train_y.min().item() - y_new.item()
        regrets.append(regret)

    return train_x, train_y, regrets


# Plotting results
def plot_results(train_x, train_y, regrets):
    # Plot optimization problem (2D parameter space)
    plt.figure()
    plt.scatter(train_x[:, 0].numpy(), train_x[:, 1].numpy(), c=-train_y.squeeze().numpy(), cmap='viridis')
    plt.title('Optimization Problem - Parameter Space')
    plt.xlabel('Learning Rate')
    plt.ylabel('Momentum')
    plt.colorbar(label='Accuracy')
    plt.show()

    # Plot cumulative regret
    cumulative_regret = np.cumsum(regrets)
    plt.figure()
    plt.plot(cumulative_regret)
    plt.title('Cumulative Regret')
    plt.xlabel('Iterations')
    plt.ylabel('Cumulative Regret')
    plt.show()

    # Contour plot near optimal solution
    plt.figure()
    x1, x2 = torch.meshgrid([torch.linspace(bounds[0, 0], bounds[1, 0], 100),
                             torch.linspace(bounds[0, 1], bounds[1, 1], 100)])
    y = torch.tensor([problem(torch.tensor([i, j])) for i, j in zip(x1.flatten(), x2.flatten())])
    y = y.view(x1.shape)
    plt.contourf(x1.numpy(), x2.numpy(), -y.numpy(), levels=50, cmap='viridis')
    plt.title('Contour Plot near Optimal Solution')
    plt.colorbar(label='Accuracy')
    plt.xlabel('Learning Rate')
    plt.ylabel('Momentum')
    plt.show()


# Main function to run the code
def main():
    train_loader, test_loader = load_data()
    problem = MPIImageClassificationProblem(train_loader, test_loader)

    bounds = torch.tensor([[0.001, 0.1], [0.9, 0.999]])  # Learning rate and momentum range
    train_x, train_y, regrets = bayesian_optimization(problem, bounds)

    # Plot the results
    plot_results(train_x, train_y, regrets)


if __name__ == "__main__":
    main()
