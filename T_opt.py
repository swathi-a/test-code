import numpy as np
import matplotlib.pyplot as plt
import optuna
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.test_functions import Branin

# Define the Branin function
problem = Branin(negate=True)  # Negate to treat as minimization problem
bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]])  # Bounds for x1 and x2
optimal_value = 0.397887  # Known global minimum of the Branin function
optimal_point = torch.tensor([9.42478, 2.475])  # Optimal point for Branin function

# TPE Optimization using Optuna
def tpe_optimization(n_trials=20):
    regrets = []  # Track regret at each iteration
    cumulative_regrets = []  # Track cumulative regret

    def objective(trial):
        x1 = trial.suggest_uniform('x1', bounds[0, 0].item(), bounds[1, 0].item())
        x2 = trial.suggest_uniform('x2', bounds[0, 1].item(), bounds[1, 1].item())
        value = problem(torch.tensor([x1, x2])).item()

        # Regret calculation
        best_value_so_far = min([optimal_value] + [trial.value for trial in trial.study.trials])
        regret = best_value_so_far - optimal_value
        regrets.append(regret)
        cumulative_regrets.append(np.sum(regrets))

        return value

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    X_samples = np.array([[trial.params['x1'], trial.params['x2']] for trial in study.trials])
    y_samples = np.array([trial.value for trial in study.trials])

    return X_samples, y_samples, study, regrets, cumulative_regrets

# Fit SingleTaskGP using BoTorch and GPyTorch
def fit_gp_model(train_x, train_y):
    # Convert to torch tensors
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32).unsqueeze(-1)

    # Define the GP model
    model = SingleTaskGP(train_x, train_y)

    # Define the Marginal Log Likelihood (MLL)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # Fit the model by maximizing the MLL
    fit_gpytorch_model(mll)

    return model

# Plot the true Branin function
def plot_branin_function():
    x1_range = torch.linspace(bounds[0, 0].item(), bounds[1, 0].item(), 100)
    x2_range = torch.linspace(bounds[0, 1].item(), bounds[1, 1].item(), 100)
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

# Plot estimated mean and variance using BoTorch GP model
def plot_estimated_mean_variance(model):
    model.eval()

    x1_range = torch.linspace(bounds[0, 0].item(), bounds[1, 0].item(), 100)
    x2_range = torch.linspace(bounds[0, 1].item(), bounds[1, 1].item(), 100)
    X1, X2 = torch.meshgrid(x1_range, x2_range)
    X_grid = torch.cat([X1.reshape(-1, 1), X2.reshape(-1, 1)], dim=1)

    with torch.no_grad():
        posterior = model.posterior(X_grid)
        mean = posterior.mean.squeeze(-1).reshape(X1.shape).numpy()
        variance = posterior.variance.squeeze(-1).reshape(X1.shape).numpy()

    # Plot GP mean
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1.numpy(), X2.numpy(), mean, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Posterior Mean')
    plt.title('BoTorch GP Posterior Mean')
    plt.show()

    # Plot GP variance
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1.numpy(), X2.numpy(), variance, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Posterior Variance')
    plt.title('BoTorch GP Posterior Variance')
    plt.show()

# Plot regret and cumulative regret
def plot_regret_cumulative_regret(regrets, cumulative_regrets):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(regrets, label='Regret', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Regret')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(cumulative_regrets, label='Cumulative Regret', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Regret')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot contour with sampled points
def plot_contour_with_sampled_points(X_samples, optimal_point):
    x1_range = torch.linspace(bounds[0, 0].item(), bounds[1, 0].item(), 100)
    x2_range = torch.linspace(bounds[0, 1].item(), bounds[1, 1].item(), 100)
    X1, X2 = torch.meshgrid(x1_range, x2_range)
    X_grid = torch.cat([X1.reshape(-1, 1), X2.reshape(-1, 1)], dim=1)
    Z = problem(X_grid).reshape(X1.shape).detach().numpy()

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1.numpy(), X2.numpy(), Z, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Branin function value')

    # Scatter the sampled points
    plt.scatter(X_samples[:, 0], X_samples[:, 1], color='red', edgecolor='black', label='Sampled Points')

    # Plot the optimal point
    plt.scatter(optimal_point[0].item(), optimal_point[1].item(), color='yellow', marker='*', s=200, label='Optimal Point')

    plt.title('Contour Plot with Sampled Points and Optimal Point')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

# Main function to run TPE optimization and plot results
def main():
    # Step 1: Plot the true Branin function
    plot_branin_function()

    # Step 2: Perform TPE optimization using Optuna
    n_trials = 50
    X_samples, y_samples, study, regrets, cumulative_regrets = tpe_optimization(n_trials=n_trials)

    # Step 3: Fit a BoTorch GP model using the TPE results
    gp_model = fit_gp_model(X_samples, y_samples)

    # Step 4: Plot estimated mean and variance using the fitted BoTorch GP model
    plot_estimated_mean_variance(gp_model)

    # Step 5: Plot regret and cumulative regret
    plot_regret_cumulative_regret(regrets, cumulative_regrets)

    # Step 6: Plot contour with sampled points and the optimal point
    plot_contour_with_sampled_points(X_samples, optimal_point)

if __name__ == "__main__":
    main()
