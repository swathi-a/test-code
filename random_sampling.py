import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Define the Branin function
def branin(x1, x2):
    a = 1.0
    b = 5.1 / (4.0 * np.pi ** 2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

# Random Sampling
def random_sampling(n_samples, bounds, optimal_value):
    x1_samples = np.random.uniform(bounds[0][0], bounds[0][1], n_samples)
    x2_samples = np.random.uniform(bounds[1][0], bounds[1][1], n_samples)
    X_samples = np.vstack((x1_samples, x2_samples)).T
    y_samples = np.array([branin(x1, x2) for x1, x2 in X_samples])
    
    # Calculate regrets
    regrets = y_samples - optimal_value
    cumulative_regrets = np.cumsum(regrets)
    
    return X_samples, y_samples, regrets, cumulative_regrets

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

# Plot regret and cumulative regret
def plot_regret_cumulative_regret(regrets, cumulative_regrets):
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

# Plot contour for Branin function and optimal points
def plot_contour_with_random_points(X_samples, bounds, optimal_point):
    x1_range = np.linspace(bounds[0][0], bounds[0][1], 100)
    x2_range = np.linspace(bounds[1][0], bounds[1][1], 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = branin(X1, X2)

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Branin function value')

    # Scatter the random sampling points
    plt.scatter(X_samples[:, 0], X_samples[:, 1], color='red', edgecolor='black', label='Random Sampling Points')

    # Plot the optimal point
    plt.scatter(optimal_point[0], optimal_point[1], color='yellow', marker='*', s=200, label='Optimal Point')

    plt.title('Contour Plot with Random Sampling Points and Optimal Point')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

# Main function
def main():
    bounds = [[-5, 10], [0, 15]]  # Bounds for x1 and x2
    optimal_value = 0.397887  # Known global minimum of the Branin function
    optimal_point = np.array([np.pi, 2.275])  # Optimal point for Branin function

    # Step 1: Plot the true Branin function
    plot_branin_function(bounds)

    # Step 2: Random sampling
    n_samples = 30
    X_samples, y_samples, regrets, cumulative_regrets = random_sampling(n_samples, bounds, optimal_value)

    # Step 3: Fit a Gaussian Process model
    gp = fit_gaussian_process(X_samples, y_samples)

    # Step 4: Plot estimated mean and variance
    plot_estimated_mean_variance(gp, bounds)

    # Step 5: Plot regret and cumulative regret
    plot_regret_cumulative_regret(regrets, cumulative_regrets)

    # Step 6: Plot contour with random sampling points and optimal point
    plot_contour_with_random_points(X_samples, bounds, optimal_point)

if __name__ == "__main__":
    main()
