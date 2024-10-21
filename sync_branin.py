import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.test_functions import SyntheticTestFunction


# Define the Branin problem inheriting from SyntheticTestFunction
class BraninProblem(SyntheticTestFunction):
    def __init__(self):
        super().__init__(noise_std=None)
        self.bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]])  # Bounds for x1 and x2

    def forward(self, x):
        x1, x2 = x[0], x[1]
        term1 = (x2 - 5.1 * (x1 ** 2) / (4 * np.pi ** 2) + 5 * x1 / np.pi - 6) ** 2
        term2 = 10 * (1 - 1 / (8 * np.pi)) * torch.cos(x1)
        return term1 + term2 + 10


# Perform Bayesian optimization on the Branin problem
def bayesian_optimization(problem, n_iter=20, n_initial_points=5):
    try:
        # Define bounds
        bounds = problem.bounds
        regrets = []
        cumulative_regrets = []

        # Generate initial random points
        train_x = torch.rand(n_initial_points, 2) * (bounds[1] - bounds[0]) + bounds[0]
        train_y = torch.stack([problem(x) for x in train_x])

        # Track regret
        optimal_value = 0.397887  # Known global minimum of the Branin function
        for y in train_y:
            regret = abs(optimal_value - y.item())
            regrets.append(regret)
            cumulative_regrets.append(sum(regrets))

        # Train the GP model
        gp = SingleTaskGP(train_x, train_y.unsqueeze(-1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        for iteration in range(n_iter):
            # Upper Confidence Bound acquisition function
            UCB = UpperConfidenceBound(gp, beta=0.1)

            # Optimize the acquisition function
            candidate, _ = optimize_acqf(
                UCB, bounds=bounds.unsqueeze(0), q=1, num_restarts=5, raw_samples=20
            )

            # Evaluate the candidate
            new_y = problem(candidate.squeeze(0))
            train_x = torch.cat([train_x, candidate], dim=0)
            train_y = torch.cat([train_y, new_y.unsqueeze(0)], dim=0)

            # Update the model
            gp.set_train_data(train_x, train_y.unsqueeze(-1), strict=False)
            fit_gpytorch_model(mll)

            # Track regret
            regret = abs(optimal_value - new_y.item())
            regrets.append(regret)
            cumulative_regrets.append(sum(regrets))

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
    # Instantiate the Branin problem
    branin_problem = BraninProblem()

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
        final_value = branin_problem(best_params)
        print(f"Best Parameters: x1 = {best_params[0].item()}, x2 = {best_params[1].item()}")
        print(f"Final value at best parameters: {final_value.item()}")
    else:
        print("Failed to find the best parameters.")


# Run the main function
if __name__ == "__main__":
    main()
