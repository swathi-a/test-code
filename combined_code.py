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
from botorch.test_functions import Branin
import optuna

# Define the Branin function
def branin(x1, x2):
    a = 1.0
    b = 5.1 / (4.0 * np.pi ** 2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

problem = Branin(negate=True)
bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]])  # Bounds for x1 and x2
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

# Bayesian Optimization
def bayesian_optimization(n_iter=20, n_initial_points=5):
    train_x = torch.rand(n_initial_points, 2) * (bounds[1] - bounds[0]) + bounds[0]
    train_y = problem(train_x).unsqueeze(-1)
    regrets = []

    for i in range(n_iter):
        print(f"Bayesian Optimization - Iteration {i + 1}")
        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        UCB = UpperConfidenceBound(model, beta=0.1)

        candidate, _ = optimize_acqf(
            UCB,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=100
        )
        
        new_x = candidate
        new_y = problem(new_x).unsqueeze(-1)

        train_x = torch.cat([train_x, new_x], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)

        best_value_so_far = train_y.min().item()
        regret = best_value_so_far - optimal_value
        regrets.append(regret)

    return train_x, train_y, regrets

# TPE Optimization
def tpe_optimization(n_trials=20):
    regrets = []
    cumulative_regrets = []

    def objective(trial):
        x1 = trial.suggest_uniform('x1', bounds[0, 0].item(), bounds[1, 0].item())
        x2 = trial.suggest_uniform('x2', bounds[0, 1].item(), bounds[1, 1].item())
        value = problem(torch.tensor([x1, x2])).item()

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

# Fit Gaussian Process for Random Sampling
def fit_gaussian_process(X_samples, y_samples):
    kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X_samples, y_samples)
    return gp

# Plot estimated mean and variance from GP
def plot_estimated_mean_variance(gp, bounds):
    x1_range = np.linspace(bounds[0, 0], bounds[1, 0], 100)
    x2_range = np.linspace(bounds[0, 1], bounds[1, 1], 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    X_grid = np.vstack([X1.ravel(), X2.ravel()]).T
    y_mean, y_std = gp.predict(X_grid, return_std=True)
    y_mean = y_mean.reshape(X1.shape)
    y_std = y_std.reshape(X1.shape)

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1, X2, y_mean, levels=50, cmap='coolwarm')
    plt.colorbar(cp, label='Estimated Mean')
    plt.title('Estimated Mean from GP')
    plt.show()

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1, X2, y_std, levels=50, cmap='plasma')
    plt.colorbar(cp, label='Estimated Standard Deviation')
    plt.title('Estimated Variance from GP')
    plt.show()

# Plot regret and cumulative regret
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
    plt.show()

# Plot contour with sampled points and optimal point
def plot_contour_with_sampled_points(X_samples, method_name):
    x1_range = np.linspace(bounds[0, 0], bounds[1, 0], 100)
    x2_range = np.linspace(bounds[0, 1], bounds[1, 1], 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = problem(torch.tensor(np.c_[X1.ravel(), X2.ravel()])).reshape(X1.shape).detach().numpy()

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Branin function value')

    plt.scatter(X_samples[:, 0], X_samples[:, 1], color='red', edgecolor='black', label=f'{method_name} Sampled Points')
    plt.scatter(optimal_point[0], optimal_point[1], color='yellow', marker='*', s=200, label='Optimal Point')
    plt.title(f'Contour Plot with {method_name} Points')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

# Main function to run all methods and compare results
def main():
    n_iterations = 20

    # Step 1: Plot the true Branin function
    plot_branin_function()

    # Random Sampling
    print("Running Random Sampling...")
    X_random, y_random, random_regrets, random_cumulative_regrets = random_sampling(n_iterations, bounds.numpy(), optimal_value)
    gp_random = fit_gaussian_process(X_random, y_random)
    plot_estimated_mean_variance(gp_random, bounds.numpy())
    plot_regret_cumulative_regret(random_regrets, random_cumulative_regrets, 'Random Sampling')
    plot_contour_with_sampled_points(X_random, 'Random Sampling')

    # Bayesian Optimization
    print("Running Bayesian Optimization...")
    train_x, train_y, bayesian_regrets = bayesian_optimization(n_iterations)
    model_bayes = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(model_bayes.likelihood, model_bayes)
    fit_gpytorch_model(mll)
    plot_estimated_mean_variance(model_bayes, bounds.numpy())
    plot_regret_cumulative_regret(bayesian_regrets, np.cumsum(bayesian_regrets), 'Bayesian Optimization')
    plot_contour_with_sampled_points(train_x.numpy(), 'Bayesian Optimization')

    # TPE Optimization
    print("Running TPE Optimization...")
    X_tpe, y_tpe, tpe_regrets, tpe_cumulative_regrets = tpe_optimization(n_iterations)
    gp_tpe = fit_gaussian_process(X_tpe, y_tpe)
    plot_estimated_mean_variance(gp_tpe, bounds.numpy())
    plot_regret_cumulative_regret(tpe_regrets, tpe_cumulative_regrets, 'TPE Optimization')
    plot_contour_with_sampled_points(X_tpe, 'TPE Optimization')

if __name__ == "__main__":
    main()
