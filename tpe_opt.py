import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import branin  # Predefined Branin function from scipy
import optuna

# Branin known global minimum point and value
optimal_value = 0.397887
optimal_point = np.array([np.pi, 2.275])

# Objective function for Optuna
def objective(trial):
    x1 = trial.suggest_uniform('x1', -5.0, 10.0)
    x2 = trial.suggest_uniform('x2', 0.0, 15.0)
    return branin(x1, x2)

# Fit Gaussian Process
def fit_gaussian_process(X_samples, y_samples):
    # Define the kernel: Constant kernel multiplied by RBF kernel
    kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
    
    # Create and fit the GP model
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X_samples, y_samples)
    
    return gp

# Plot the true Branin function
def plot_branin_function(bounds):
    x1_range = np.linspace(bounds[0][0], bounds[0][1], 100)
    x2_range = np.linspace(bounds[1][0], bounds[1][1], 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = branin(X1, X2)
    
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Branin function value')
    plt.title('True Branin Function')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# Plot estimated mean and variance
def plot_estimated_mean_variance(gp, bounds):
    # Create a grid for prediction
    x1_range = np.linspace(bounds[0][0], bounds[0][1], 100)
    x2_range = np.linspace(bounds[1][0], bounds[1][1], 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    X_grid = np.vstack([X1.ravel(), X2.ravel()]).T

    # Predict the mean and standard deviation
    y_mean, y_std = gp.predict(X_grid, return_std=True)
    y_mean = y_mean.reshape(X1.shape)
    y_std = y_std.reshape(X1.shape)

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
    cp = plt.contourf(X1, X2, y_std, levels=50, cmap='plasma')
    plt.colorbar(cp, label='Estimated Standard Deviation')
    plt.title('Estimated Standard Deviation (Variance) from GP')
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
    Z = branin(X1, X2)

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Branin function value')

    # Plot random sampled points
    plt.scatter(X_samples[:, 0], X_samples[:, 1], label="TPE Sampling Points", color="red", edgecolor="k")

    # Highlight the optimal point
    plt.scatter(optimal_point[0], optimal_point[1], marker="*", color="yellow", s=200, label="Optimal Point")
    
    plt.title("Contour Plot with TPE Sampling Points and Optimal Point")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

# Main function to run optimization and plot results
def main():
    bounds = [[-5, 10], [0, 15]]  # Bounds for x1 and x2
    
    # Step 1: Plot the true Branin function
    plot_branin_function(bounds)

    # Step 2: Optimize using Optuna's TPE sampler
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='minimize')
    study.optimize(objective, n_trials=30)

    # Extract the sampled points and values from the study
    X_samples = np.array([[trial.params['x1'], trial.params['x2']] for trial in study.trials])
    y_samples = np.array([trial.value for trial in study.trials])

    # Step 3: Fit a Gaussian Process model to the TPE sampled points
    gp = fit_gaussian_process(X_samples, y_samples)

    # Step 4: Plot estimated mean and variance based on the fitted model
    plot_estimated_mean_variance(gp, bounds)

    # Step 5: Calculate and plot regret and cumulative regret
    regrets, cumulative_regret = calculate_regret(y_samples, optimal_value)
    plot_regret(regrets, cumulative_regret)

    # Step 6: Plot contour with optimal point and TPE sampled points
    plot_contour_with_samples(X_samples, bounds, optimal_point)

if __name__ == "__main__":
    main()
