import numpy as np
import torch
import optuna
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.optimize import branin
import traceback

# Define the known Branin global minimum and value
optimal_value = 0.397887
optimal_point = np.array([np.pi, 2.275])

# Branin function
def branin_function(x1, x2):
    return branin(x1, x2)

# TPE Optimization using Optuna
def tpe_optimization(n_trials=50):
    try:
        def objective(trial):
            x1 = trial.suggest_uniform('x1', -5.0, 10.0)
            x2 = trial.suggest_uniform('x2', 0.0, 15.0)
            y = branin_function(x1, x2)
            return y

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=n_trials)

        # Gather the samples and values
        x1_samples = np.array([trial.params['x1'] for trial in study.trials])
        x2_samples = np.array([trial.params['x2'] for trial in study.trials])
        X_samples = np.vstack((x1_samples, x2_samples)).T
        y_samples = np.array([trial.value for trial in study.trials])

        return X_samples, y_samples, study
    except Exception as e:
        print(f"Error during TPE optimization: {e}")
        traceback.print_exc()
        return np.empty((0, 2)), np.empty(0)

# Fit Gaussian Process model using BoTorch
def fit_gp_model(X_samples, y_samples):
    try:
        # Convert data to tensors
        X_train = torch.tensor(X_samples, dtype=torch.float32)
        y_train = torch.tensor(y_samples.reshape(-1, 1), dtype=torch.float32)

        # Define and fit GP model
        model = SingleTaskGP(X_train, y_train, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        return model
    except Exception as e:
        print(f"Error during GP model fitting: {e}")
        traceback.print_exc()
        return None

# Plot the true Branin function
def plot_branin_function(bounds):
    x1_range = np.linspace(bounds[0][0], bounds[0][1], 100)
    x2_range = np.linspace(bounds[1][0], bounds[1][1], 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = branin_function(X1, X2)
    
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Branin function value')
    plt.title('True Branin Function')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# Plot estimated mean and variance using GP model
def plot_estimated_mean_variance(gp_model, bounds):
    # Create a grid for prediction
    x1_range = np.linspace(bounds[0][0], bounds[0][1], 100)
    x2_range = np.linspace(bounds[1][0], bounds[1][1], 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    X_grid = np.vstack([X1.ravel(), X2.ravel()]).T
    X_grid_tensor = torch.tensor(X_grid, dtype=torch.float32)

    # Predict the mean and variance
    y_mean, y_var = gp_model.posterior(X_grid_tensor).mean.detach().numpy(), gp_model.posterior(X_grid_tensor).variance.detach().numpy()
    y_mean = y_mean.reshape(X1.shape)
    y_var = y_var.reshape(X1.shape)

    # Plot the mean
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1, X2, y_mean, levels=50, cmap='coolwarm')
    plt.colorbar(cp, label='Estimated Mean')
    plt.title('Estimated Mean from GP')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

    # Plot the variance
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1, X2, y_var, levels=50, cmap='plasma')
    plt.colorbar(cp, label='Estimated Variance')
    plt.title('Estimated Variance from GP')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# Calculate regret and cumulative regret
def calculate_regret(y_samples, optimal_value):
    regrets = y_samples - optimal_value
    cumulative_regret = np.cumsum(regrets)
    return regrets, cumulative_regret

# Plot regret and cumulative regret
def plot_regret(regrets, cumulative_regret):
    plt.figure(figsize=(8, 6))
    plt.plot(regrets, label="Regret")
    plt.xlabel("Iteration")
    plt.ylabel("Regret")
    plt.title("Regret Over Iterations")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(cumulative_regret, label="Cumulative Regret", color="orange")
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret Over Iterations")
    plt.legend()
    plt.show()

# Plot contour and check proximity to the optimal point
def plot_contour_with_samples(X_samples, bounds, optimal_point):
    x1_range = np.linspace(bounds[0][0], bounds[0][1], 100)
    x2_range = np.linspace(bounds[1][0], bounds[1][1], 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = branin_function(X1, X2)

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Branin function value')

    # Plot random sampled points
    plt.scatter(X_samples[:, 0], X_samples[:, 1], label="TPE Sampled Points", color="red", edgecolor="k")

    # Highlight the optimal point
    plt.scatter(optimal_point[0], optimal_point[1], marker="*", color="yellow", s=200, label="Optimal Point")
    
    plt.title("Contour Plot with TPE Sampled Points and Optimal Point")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

# Main function to run the optimization and plot the results
def main():
    bounds = [[-5, 10], [0, 15]]  # Bounds for x1 and x2
    
    # Step 1: Plot the true Branin function
    plot_branin_function(bounds)

    # Step 2: Perform TPE optimization using Optuna
    n_trials = 50
    X_samples, y_samples, study = tpe_optimization(n_trials=n_trials)

    # Step 3: Fit a Gaussian Process model using BoTorch
    gp_model = fit_gp_model(X_samples, y_samples)

    # Step 4: Plot estimated mean and variance using the fitted GP model
    plot_estimated_mean_variance(gp_model, bounds)

    # Step 5: Calculate and plot regret and cumulative regret
    regrets, cumulative_regret = calculate_regret(y_samples, optimal_value)
    plot_regret(regrets, cumulative_regret)

    # Step 6: Plot contour with optimal point and sampled points
    plot_contour_with_samples(X_samples, bounds, optimal_point)

if __name__ == "__main__":
    main()
