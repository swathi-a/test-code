import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.test_functions import SyntheticTestFunction
from gpytorch.mlls import ExactMarginalLogLikelihood
import math

# Custom Branin problem inheriting from SyntheticTestFunction
class Branin(SyntheticTestFunction):
    r"""Branin test function.
    
    Two-dimensional function (usually evaluated on [-5, 10] x [0, 15]):
    
    B(x) = (x_2 - b x_1^2 + c x_1 - r)^2 + 10 (1 - t) cos(x_1) + 10
    
    Where `b`, `c`, `r`, and `t` are constants:
        b = 5.1 / (4 * math.pi**2)
        c = 5 / math.pi
        r = 6
        t = 1 / (8 * math.pi)
    
    Global minimizers:
        z_1 = (-pi, 12.275), z_2 = (pi, 2.275), z_3 = (9.42478, 2.475)
    with B(z_i) = 0.397887
    """
    
    dim = 2
    _bounds = [(-5.0, 10.0), (0.0, 15.0)]  # x1 and x2 bounds
    _optimal_value = 0.397887
    _optimizers = [(-math.pi, 12.275), (math.pi, 2.275), (9.42478, 2.475)]  # Known minima
    
    def __init__(self, noise_std=None, negate=False):
        super().__init__(noise_std=noise_std, negate=negate, bounds=self._bounds)
    
    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        t1 = (
            X[..., 1]
            - 5.1 / (4 * math.pi ** 2) * X[..., 0].pow(2)
            + 5 / math.pi * X[..., 0]
            - 6
        )
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X[..., 0])
        return t1.pow(2) + t2 + 10

# Instantiate the Branin function
problem = Branin(negate=True)  # Set to "negate=True" to treat it as a minimization problem
bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]])  # Bounds for x1 and x2
optimal_value = 0.397887  # Known global minimum of the Branin function
optimal_point = torch.tensor([9.42478, 2.475])  # Optimal point for Branin function

# Bayesian Optimization function
def bayesian_optimization(n_iter=20, n_initial_points=5):
    try:
        train_x = torch.rand(n_initial_points, 2) * (bounds[1] - bounds[0]) + bounds[0]  # Initial random points
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
            regret = best_value_so_far - optimal_value
            regrets.append(regret)

        return train_x, train_y, regrets

    except Exception as e:
        print(f"Error during Bayesian optimization: {e}")
        return None, None, None

# Plot the true Branin function
def plot_branin_function(bounds):
    try:
        x1_range = np.linspace(bounds[0][0].item(), bounds[1][0].item(), 100)
        x2_range = np.linspace(bounds[0][1].item(), bounds[1][1].item(), 100)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        X_grid = np.vstack([X1.ravel(), X2.ravel()]).T
        Z = problem(torch.tensor(X_grid, dtype=torch.float32)).reshape(X1.shape).detach().numpy()

        plt.figure(figsize=(8, 6))
        cp = plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
        plt.colorbar(cp, label='Branin function value')
        plt.title('True Branin Function')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

    except Exception as e:
        print(f"Error in plotting Branin function: {e}")

# Plot estimated mean and variance from the GP model
def plot_estimated_mean_variance(model, bounds):
    try:
        # Create a grid for prediction
        x1_range = torch.linspace(bounds[0][0].item(), bounds[1][0].item(), 100)
        x2_range = torch.linspace(bounds[0][1].item(), bounds[1][1].item(), 100)
        X1, X2 = torch.meshgrid(x1_range, x2_range)
        X_grid = torch.cat([X1.reshape(-1, 1), X2.reshape(-1, 1)], dim=1)

        # Predict the mean and variance
        model.eval()
        with torch.no_grad():
            y_mean, y_var = model(X_grid).mean, model(X_grid).variance
            y_mean = y_mean.reshape(X1.shape).detach().numpy()
            y_std = torch.sqrt(y_var).reshape(X1.shape).detach().numpy()

        # Plot the mean
        plt.figure(figsize=(8, 6))
        cp = plt.contourf(X1.numpy(), X2.numpy(), y_mean, levels=50, cmap='coolwarm')
        plt.colorbar(cp, label='Estimated Mean')
        plt.title('Estimated Mean from GP')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

        # Plot the variance
        plt.figure(figsize=(8, 6))
        cp = plt.contourf(X1.numpy(), X2.numpy(), y_std, levels=50, cmap='plasma')
        plt.colorbar(cp, label='Estimated Standard Deviation')
        plt.title('Estimated Standard Deviation (Variance) from GP')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

    except Exception as e:
        print(f"Error in plotting estimated mean and variance: {e}")

# Plot regret and cumulative regret
def plot_regret_cumulative_regret(regrets):
    try:
        cumulative_regrets = np.cumsum(regrets)

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

    except Exception as e:
        print(f"Error in plotting regret: {e}")

# Plot contour with sampled points and optimal point
def plot_contour_with_sampled_points(train_x, bounds, optimal_point):
    try:
        x1_range = np.linspace(bounds[0][0].item(), bounds[1][0].item(), 100)
        x2_range = np.linspace(bounds[0][1].item(), bounds[1][1].item(), 100)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        X_grid = np.vstack([X1.ravel(), X2.ravel()]).T
        Z = problem(torch.tensor(X_grid, dtype=torch.float32)).reshape(X1.shape).detach().numpy()

        plt.figure(figsize=(8, 6))
        cp = plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
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

    except Exception as e:
        print(f"Error in plotting contour with sampled points: {e}")

# Main function
def main():
    try:
        n_iterations = 20

        # Step 1: Plot the true Branin function
        plot_branin_function(bounds)

        # Step 2: Bayesian optimization
        train_x, train_y, regrets = bayesian_optimization(n_iter=n_iterations)

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

    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
