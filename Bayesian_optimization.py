import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.test_functions import Branin
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
import matplotlib.pyplot as plt

# Define the Branin function and known optimal value
problem = Branin(negate=True)  # Set to negate=True because BoTorch maximizes by default
bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]])
optimal_value = 0.397887  # Known global minimum
optimal_point = torch.tensor([np.pi, 2.275])  # Optimal point

# Bayesian Optimization using BoTorch
def bayesian_optimization(n_iter=20, n_initial_points=5):
    # Generate initial random points
    train_x = torch.rand(n_initial_points, 2) * (bounds[1] - bounds[0]) + bounds[0]
    train_y = problem(train_x).unsqueeze(-1)

    regrets = []
    
    for i in range(n_iter):
        # Fit a GP model
        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        # Define the acquisition function (UCB)
        UCB = UpperConfidenceBound(model, beta=0.1)
        
        # Optimize the acquisition function to find the next point
        candidate, _ = optimize_acqf(
            UCB,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20
        )

        # Evaluate the Branin function at the new candidate point
        new_y = problem(candidate).unsqueeze(-1)
        
        # Update training points
        train_x = torch.cat([train_x, candidate])
        train_y = torch.cat([train_y, new_y])
        
        # Compute the regret
        best_value_so_far = train_y.min().item()
        regret = best_value_so_far - optimal_value
        regrets.append(regret)

    return train_x, train_y, regrets

# Function to calculate cumulative regret
def calculate_cumulative_regret(regrets):
    return np.cumsum(regrets)

# Plot the true Branin function
def plot_branin_function():
    x1 = np.linspace(-5, 10, 100)
    x2 = np.linspace(0, 15, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = problem(torch.tensor([X1, X2])).numpy().reshape(100, 100)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Branin Function Value')
    plt.title('True Branin Function')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# Plot estimated mean and variance using BoTorch GP model
def plot_estimated_mean_variance(model, bounds):
    # Create a grid for prediction
    x1 = np.linspace(bounds[0, 0].item(), bounds[1, 0].item(), 100)
    x2 = np.linspace(bounds[0, 1].item(), bounds[1, 1].item(), 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = torch.tensor(np.vstack([X1.ravel(), X2.ravel()]).T)

    # Predict the mean and variance (standard deviation)
    model.eval()
    with torch.no_grad():
        pred_mean, pred_var = model(X_grid).mean, model(X_grid).variance
        pred_mean = pred_mean.reshape(X1.shape).numpy()
        pred_var = pred_var.reshape(X1.shape).numpy()

    # Plot the mean
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1, X2, pred_mean, levels=50, cmap='coolwarm')
    plt.colorbar(cp, label='Estimated Mean')
    plt.title('Estimated Mean from GP')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

    # Plot the variance
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1, X2, np.sqrt(pred_var), levels=50, cmap='plasma')
    plt.colorbar(cp, label='Estimated Standard Deviation')
    plt.title('Estimated Standard Deviation (Variance) from GP')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# Plot the regret and cumulative regret
def plot_regret_cumulative_regret(regrets):
    cumulative_regrets = calculate_cumulative_regret(regrets)

    plt.figure(figsize=(10, 6))
    
    # Plot regret
    plt.subplot(2, 1, 1)
    plt.plot(regrets, label='Regret')
    plt.title('Regret per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Regret')
    plt.legend()

    # Plot cumulative regret
    plt.subplot(2, 1, 2)
    plt.plot(cumulative_regrets, label='Cumulative Regret', color='orange')
    plt.title('Cumulative Regret over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Regret')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot the contour with sampled points and optimal point
def plot_contour_with_sampled_points(X_samples, bounds, optimal_point):
    x1 = np.linspace(bounds[0, 0].item(), bounds[1, 0].item(), 100)
    x2 = np.linspace(bounds[0, 1].item(), bounds[1, 1].item(), 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = problem(torch.tensor([X1, X2])).numpy().reshape(100, 100)

    plt.figure(figsize=(8, 6))
    plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Branin function value')

    # Scatter the sampled points
    plt.scatter(X_samples[:, 0].numpy(), X_samples[:, 1].numpy(), color='red', edgecolor='black', label='Sampled Points')

    # Plot the optimal point
    plt.scatter(optimal_point[0].item(), optimal_point[1].item(), color='yellow', marker='*', s=200, label='Optimal Point')

    plt.title('Contour Plot with Sampled Points and Optimal Point')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

# Main function to run Bayesian Optimization and plotting
def main():
    n_iterations = 20

    # Step 1: Plot the true Branin function
    plot_branin_function()

    # Step 2: Perform Bayesian Optimization
    X_samples, Y_samples, regrets = bayesian_optimization(n_iter=n_iterations)

    # Step 3: Fit the final GP model to the data
    model = SingleTaskGP(X_samples, Y_samples)
    fit_gpytorch_model(ExactMarginalLogLikelihood(model.likelihood, model))

    # Step 4: Plot the estimated mean and variance
    plot_estimated_mean_variance(model, bounds)

    # Step 5: Plot regret and cumulative regret
    plot_regret_cumulative_regret(regrets)

    # Step 6: Plot contour with sampled points and optimal point
    plot_contour_with_sampled_points(X_samples, bounds, optimal_point)

if __name__ == "__main__":
    main()
