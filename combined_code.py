import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.mlls import ExactMarginalLogLikelihood
from botorch.test_functions.branin import Branin
import optuna
import os

# Set up the directory to save plots
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

# Define the Branin function for grid and random sampling
def branin(x1, x2):
    a = 1.0
    b = 5.1 / (4.0 * np.pi ** 2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

# Set bounds and optimal value for Branin
bounds = [[-5.0, 0.0], [10.0, 15.0]]  # Bounds for x1 and x2
optimal_value = 0.397887  # Known optimal value
optimal_point = np.array([np.pi, 2.275])  # Approximate optimal point for Branin

# Random Sampling
def random_sampling(n_samples, bounds, optimal_value):
    x1_samples = np.random.uniform(bounds[0][0], bounds[0][1], n_samples)
    x2_samples = np.random.uniform(bounds[1][0], bounds[1][1], n_samples)
    X_samples = np.vstack((x1_samples, x2_samples)).T
    y_samples = np.array([branin(x1, x2) for x1, x2 in X_samples])
    
    regrets = y_samples - optimal_value
    cumulative_regrets = np.cumsum(regrets)
    
    return X_samples, y_samples, regrets, cumulative_regrets

# Bayesian Optimization using BoTorch
def bayesian_optimization(n_iter=20, n_initial_points=5):
    # Initialize train data
    train_x = torch.rand(n_initial_points, 2) * (torch.tensor(bounds)[1] - torch.tensor(bounds)[0]) + torch.tensor(bounds)[0]
    branin_func = Branin()
    train_y = branin_func(train_x).unsqueeze(-1)  # Using BoTorch Branin for Bayesian Optimization
    regrets = []

    for i in range(n_iter):
        print(f"Bayesian Optimization - Iteration {i + 1}")
        # Fit GP model using BoTorch
        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        # Define the acquisition function
        UCB = UpperConfidenceBound(model, beta=0.1)

        # Optimize the acquisition function
        candidate, _ = optimize_acqf(
            UCB,
            bounds=torch.tensor(bounds).T,  # BoTorch expects bounds as a 2x2 tensor
            q=1,
            num_restarts=10,
            raw_samples=100
        )
        
        # Evaluate the new candidate point
        new_x = candidate
        new_y = branin_func(new_x).unsqueeze(-1)

        # Update training data
        train_x = torch.cat([train_x, new_x], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)

        # Compute regret
        best_value_so_far = train_y.min().item()
        regret = best_value_so_far - optimal_value
        regrets.append(regret)

    return train_x, train_y, regrets

# TPE Optimization using Optuna
def tpe_optimization(n_trials=20):
    regrets = []
    cumulative_regrets = []

    def objective(trial):
        x1 = trial.suggest_uniform('x1', bounds[0][0], bounds[1][0])
        x2 = trial.suggest_uniform('x2', bounds[0][1], bounds[1][1])
        value = branin(x1, x2)

        best_value_so_far = min([optimal_value] + [trial.value for trial in trial.study.trials])
        regret = best_value_so_far - optimal_value
        regrets.append(regret)
        cumulative_regrets.append(np.sum(regrets))

        return value

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    X_samples = np.array([[trial.params['x1'], trial.params['x2']] for trial in study.trials])
    y_samples = np.array([trial.value for trial in study.trials])

    return X_samples, y_samples, regrets, cumulative_regrets

# Plot functions
def plot_branin_function():
    x1_range = np.linspace(bounds[0][0], bounds[1][0], 100)
    x2_range = np.linspace(bounds[0][1], bounds[1][1], 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = np.array([branin(x1, x2) for x1, x2 in zip(X1.ravel(), X2.ravel())]).reshape(X1.shape)

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Branin function value')
    plt.title('True Branin Function')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(f"{output_dir}/branin_function.png")
    plt.show()

def fit_gaussian_process(X_samples, y_samples):
    kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X_samples, y_samples)
    return gp

def plot_estimated_mean_variance(gp, bounds, method_name):
    x1_range = np.linspace(bounds[0][0], bounds[1][0], 100)
    x2_range = np.linspace(bounds[0][1], bounds[1][1], 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    X_grid = np.vstack([X1.ravel(), X2.ravel()]).T
    y_mean, y_std = gp.predict(X_grid, return_std=True)
    y_mean = y_mean.reshape(X1.shape)
    y_std = y_std.reshape(X1.shape)

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1, X2, y_mean, levels=50, cmap='coolwarm')
    plt.colorbar(cp, label='Estimated Mean')
    plt.title(f'Estimated Mean from GP - {method_name}')
    plt.savefig(f"{output_dir}/estimated_mean_{method_name}.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1, X2, y_std, levels=50, cmap='plasma')
    plt.colorbar(cp, label='Estimated Standard Deviation')
    plt.title(f'Estimated Variance from GP - {method_name}')
    plt.savefig(f"{output_dir}/estimated_variance_{method_name}.png")
    plt.show()

def plot_regret_cumulative_regret(regrets, cumulative_regrets, method_name):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(regrets, label=f'{method_name} Regret')
    plt.title(f'{method_name} Regret per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Regret')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(cumulative_regrets, label=f'{method_name} Cumulative Regret', color='orange')
    plt.title(f'{method_name} Cumulative Regret over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/regret_cumulative_{method_name}.png")
    plt.show()

def plot_contour_with_sampled_points(X_samples, method_name):
    x1_range = np.linspace(bounds[0][0], bounds[1][0], 100)
    x2_range = np.linspace(bounds[0][1], bounds[1][1], 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = np.array([branin(x1, x2) for x1, x2 in zip(X1.ravel(), X2.ravel())]).reshape(X1.shape)

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Branin function value')

    plt.scatter(X_samples[:, 0], X_samples[:, 1], color='red', edgecolor='black', label=f'{method_name} Sampled Points')
    plt.scatter(optimal_point[0], optimal_point[1], color='yellow', marker='*', s=200, label='Optimal Point')
    plt.title(f'Contour Plot with {method_name} Points')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.savefig(f"{output_dir}/contour_{method_name}.png")
    plt.show()

# Main function to run all methods and compare results
def main():
    n_iterations = 20

    # Step 1: Plot the true Branin function
    plot_branin_function()

    # Random Sampling
    print("Running Random Sampling...")
    X_random, y_random, random_regrets, random_cumulative_regrets = random_sampling(n_iterations, bounds, optimal_value)
    gp_random = fit_gaussian_process(X_random, y_random)
    plot_estimated_mean_variance(gp_random, bounds, 'Random Sampling')
    plot_regret_cumulative_regret(random_regrets, random_cumulative_regrets, 'Random Sampling')
    plot_contour_with_sampled_points(X_random, 'Random Sampling')

    # Bayesian Optimization using BoTorch model fitting
    print("Running Bayesian Optimization with BoTorch...")
    train_x, train_y, bayesian_regrets = bayesian_optimization(n_iterations)  # BoTorch Model Fitting
    plot_estimated_mean_variance(fit_gaussian_process(train_x.numpy(), train_y.numpy()), bounds, 'Bayesian Optimization')
    plot_regret_cumulative_regret(bayesian_regrets, np.cumsum(bayesian_regrets), 'Bayesian Optimization')
    plot_contour_with_sampled_points(train_x.numpy(), 'Bayesian Optimization')

    # TPE Optimization using Optuna fitting model
    print("Running TPE Optimization with Optuna...")
    X_tpe, y_tpe, tpe_regrets, tpe_cumulative_regrets = tpe_optimization(n_iterations)  # Optuna TPE Model Fitting
    plot_regret_cumulative_regret(tpe_regrets, tpe_cumulative_regrets, 'TPE Optimization')
    plot_contour_with_sampled_points(X_tpe, 'TPE Optimization')

if __name__ == "__main__":
    main()
