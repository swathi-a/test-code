import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.test_functions import SyntheticTestFunction
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


# Bayesian Optimization on the Branin problem
def bayesian_optimization(problem, n_iter=20, n_initial_points=5):
    try:
        # Define bounds
        bounds = torch.tensor(problem.bounds).T
        regrets = []
        cumulative_regrets = []

        # Generate initial random points
        train_x = torch.rand(n_initial_points, 2) * (bounds[1] - bounds[0]) + bounds[0]
        train_y = torch.tensor([problem.evaluate_true(x.unsqueeze(0)).item() for x in train_x]).unsqueeze(-1)

        # Track regret
        optimal_value = problem._optimal_value
        for y in train_y:
            regret = abs(optimal_value - y.item())
            regrets.append(regret)
            cumulative_regrets.append(sum(regrets))

        # Train the GP model
        gp = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        for iteration in range(n_iter):
            try:
                # Upper Confidence Bound acquisition function
                UCB = UpperConfidenceBound(gp, beta=0.1)

                # Optimize the acquisition function
                candidate, _ = optimize_acqf(
                    UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20
                )

                # Evaluate the candidate
                new_y = problem.evaluate_true(candidate)
                train_x = torch.cat([train_x, candidate], dim=0)
                train_y = torch.cat([train_y, new_y.unsqueeze(-1)], dim=0)

                # Update the model
                gp.set_train_data(train_x, train_y, strict=False)
                fit_gpytorch_model(mll)

                # Track regret
                regret = abs(optimal_value - new_y.item())
                regrets.append(regret)
                cumulative_regrets.append(sum(regrets))

            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                continue

        # Get the best parameters from the search
        best_params = train_x[train_y.argmin()]
        return best_params, regrets, cumulative_regrets
    except Exception as e:
        print(f"Error during Bayesian optimization: {e}")
        return None, None, None


# Plot the actual Branin function
def plot_branin():
    x1 = np.linspace(-5, 10, 200)
    x2 = np.linspace(0, 15, 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = (X2 - 5.1 * (X1 ** 2) / (4 * np.pi ** 2) + 5 * X1 / np.pi - 6) ** 2 + \
        10 * (1 - 1 / (8 * np.pi)) * np.cos(X1) + 10

    plt.figure(figsize=(8, 6))
    plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Branin Function Value')
    plt.title('Branin Function')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


# Plot regret and cumulative regret
def plot_regret(regrets, cumulative_regrets):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(regrets, label='Regret')
    plt.xlabel('Iteration')
    plt.ylabel('Regret')
    plt.title('Regret over Iterations')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(cumulative_regrets, label='Cumulative Regret', color='orange')
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret over Iterations')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# Plot the contour graph near the optimal point
def plot_contour_near_optimal(best_params):
    x1 = np.linspace(best_params[0].item() - 0.5, best_params[0].item() + 0.5, 200)
    x2 = np.linspace(best_params[1].item() - 0.5, best_params[1].item() + 0.5, 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = (X2 - 5.1 * (X1 ** 2) / (4 * np.pi ** 2) + 5 * X1 / np.pi - 6) ** 2 + \
        10 * (1 - 1 / (8 * np.pi)) * np.cos(X1) + 10

    plt.figure(figsize=(8, 6))
    plt.contourf(X1, X2, Z, levels=50, cmap='coolwarm')
    plt.colorbar(label='Branin Function Value')
    plt.plot(best_params[0], best_params[1], 'ro', label='Optimal Point')
    plt.title('Contour near Optimal Point')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


# Main function to run the entire process
def main():
    try:
        # Instantiate the Branin problem
        branin_problem = Branin()

        # Run Bayesian optimization
        best_params, regrets, cumulative_regrets = bayesian_optimization(branin_problem)

        if best_params is not None:
            # Plot the actual Branin function
            plot_branin()

            # Plot regret and cumulative regret
            plot_regret(regrets, cumulative_regrets)

            # Plot the contour graph near the optimal point
            plot_contour_near_optimal(best_params)

            # Final evaluation with the best parameters
            final_value = branin_problem.evaluate_true(best_params.unsqueeze(0))
            print(f"Best Parameters: x1 = {best_params[0].item()}, x2 = {best_params[1].item()}")
            print(f"Final value at best parameters: {final_value.item()}")
        else:
            print("Failed to find the best parameters.")
    except Exception as e:
        print(f"Error in main function: {e}")


# Run the main function
if __name__ == "__main__":
    main()
