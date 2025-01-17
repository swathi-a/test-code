import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.test_functions import Branin
from gpytorch.mlls import ExactMarginalLogLikelihood

# Define the Branin function
problem = Branin(negate=True)  # Set to "negate=True" to treat it as a minimization problem
bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]])  # Bounds for x1 and x2
optimal_value = 0.397887  # Known global minimum of the Branin function
optimal_point = torch.tensor([9.42478, 2.475])  # Optimal point for Branin function

# Grid sampling function
def grid_sampling(n_points, bounds):
    x1_range = torch.linspace(bounds[0, 0], bounds[1, 0], int(n_points**0.5))
    x2_range = torch.linspace(bounds[0, 1], bounds[1, 1], int(n_points**0.5))
    X1, X2 = torch.meshgrid(x1_range, x2_range)
    grid_points = torch.cat([X1.reshape(-1, 1), X2.reshape(-1, 1)], dim=1)
    return grid_points

# Bayesian Optimization function
def bayesian_optimization(n_iter=20, n_initial_points=5):
    # Change from random to grid sampling
    train_x = grid_sampling(n_initial_points, bounds)
    train_y = problem(train_x).unsqueeze(-1)  # Initial evaluations of Branin function
    regrets = []

    for i in range(n_iter):
        # Step 1: Fit the Gaussian Process (GP) model
        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        # Step 2: Define the acquisition function (Upper Confidence Bound)
        UCB = UpperConfidenceBound(model, beta=0.1)

        # Step 3: Optimize the acquisition function to get the next candidate point
        candidate, _ = optimize_acqf(
            UCB,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=100
        )
        
        # Step 4: Evaluate the Branin function at the candidate point
        new_x = candidate
        new_y = problem(new_x).unsqueeze(-1)

        # Step 5: Update the training data
        train_x = torch.cat([train_x, new_x], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)

        # Step 6: Calculate the regret
        best_value_so_far = train_y.min().item()
        regret = optimal_value - best_value_so_far
        regrets.append(regret)

    return train_x, train_y, regrets

# Plotting the true Branin function
def plot_branin_function(bounds):
    x1_range = torch.linspace(bounds[0][0], bounds[1][0], 100)
    x2_range = torch.linspace(bounds[0][1], bounds[1][1], 100)
    X1, X2 = torch.meshgrid(x1_range, x2_range)
    X_grid = torch.cat([X1.reshape(-1, 1), X2.reshape(-1, 1)], dim=1)
    Z = problem(X_grid).reshape(X1.shape).detach().numpy()

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1.numpy(), X2.numpy(), Z, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Branin function value')
    plt.title('True Branin Function')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# Plot the estimated mean and variance from GP model
def plot_estimated_mean_variance(model, bounds):
    x1_range = torch.linspace(bounds[0][0], bounds[1][0], 100)
    x2_range = torch.linspace(bounds[0][1], bounds[1][1], 100)
    X1, X2 = torch.meshgrid(x1_range, x2_range)
    X_grid = torch.cat([X1.reshape(-1, 1), X2.reshape(-1, 1)], dim=1)

    with torch.no_grad():
        pred = model.posterior(X_grid).mean
        variance = model.posterior(X_grid).variance

    pred = pred.reshape(100, 100).detach().numpy()
    variance = variance.reshape(100, 100).detach().numpy()

    # Plot Estimated Mean
    plt.figure(figsize=(10, 8))
    plt.contourf(X1.numpy(), X2.numpy(), pred, levels=50, cmap='viridis')
    plt.title('Estimated Mean from GP')
    plt.colorbar()
    plt.show()

    # Plot Estimated Variance
    plt.figure(figsize=(10, 8))
    plt.contourf(X1.numpy(), X2.numpy(), variance, levels=50, cmap='viridis')
    plt.title('Estimated Standard Deviation (Variance) from GP')
    plt.colorbar()
    plt.show()

# Plot regret and cumulative regret over iterations
def plot_regret_cumulative_regret(regrets):
    cumulative_regrets = np.cumsum(regrets)

    plt.figure(figsize=(10, 6))

    # Plot regret per iteration
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

# Plot contour with sampled points and optimal point
def plot_contour_with_sampled_points(train_x, bounds, optimal_point):
    x1_range = torch.linspace(bounds[0][0], bounds[1][0], 100)
    x2_range = torch.linspace(bounds[0][1], bounds[1][1], 100)
    X1, X2 = torch.meshgrid(x1_range, x2_range)
    X_grid = torch.cat([X1.reshape(-1, 1), X2.reshape(-1, 1)], dim=1)
    Z = problem(X_grid).reshape(X1.shape).detach().numpy()

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1.numpy(), X2.numpy(), Z, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Branin function value')

    # Scatter the sampled points
    plt.scatter(train_x[:, 0].numpy(), train_x[:, 1].numpy(), color='red', edgecolor='black', label='Sampled Points')

    # Plot the optimal point
    plt.scatter(optimal_point[0], optimal_point[1], color='yellow', marker='*', s=200, label='Optimal Point')

    plt.title('Contour Plot with Sampled Points and Optimal Point')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

# Main function
def main():
    n_iterations = 20

    # Step 1: Plot the true Branin function
    plot_branin_function(bounds)

    # Step 2: Bayesian optimization with grid sampling
    train_x, train_y, regrets = bayesian_optimization(n_iter=n_iterations, n_initial_points=9)

    # Step 3: Fit the GP model using the final data
    model = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # Step 4: Plot estimated mean and variance
    plot_estimated_mean_variance(model, bounds)

    # Step 5: Plot regret and cumulative regret
    plot_regret_cumulative_regret(regrets)

    # Step 6: Plot contour with sampled points and optimal point
    plot_contour_with_sampled_points(train_x, bounds, optimal_point)

if __name__ == "__main__":
    main()
