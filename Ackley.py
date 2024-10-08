import torch
from botorch.test_functions import Ackley
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize
from botorch.models.transforms.outcome import Standardize
import matplotlib.pyplot as plt
import numpy as np
import optuna
import warnings
import traceback
from botorch.utils.multi_objective import is_non_dominated

warnings.filterwarnings("ignore")

# Define the Ackley function
problem = Ackley(negate=True)
bounds = torch.tensor([[-5.0, -5.0], [10.0, 10.0]])
optimal_value = 0.0  # Known global minimum of the Ackley function at [0, 0]
optimal_point = np.array([0.0, 0.0])  # Optimal point for Ackley function

# Random Sampling
def random_sampling(n_iter=20):
    try:
        random_x = torch.rand(n_iter, 2) * (bounds[1] - bounds[0]) + bounds[0]
        random_y = problem(random_x).unsqueeze(-1)
        regrets = []
        for i in range(n_iter):
            best_value_so_far = random_y[:i+1].min().item()
            regret = best_value_so_far - optimal_value
            regrets.append(regret)
        return random_x, random_y, regrets
    except Exception as e:
        print(f"Random search error: {e}")
        return torch.empty((0, 2)), torch.empty((0, 1)), []

# Bayesian Optimization
def bayesian_optimization(n_iter=20, n_initial_points=5):
    try:
        train_x = torch.rand(n_initial_points, 2) * (bounds[1] - bounds[0]) + bounds[0]
        train_y = problem(train_x).unsqueeze(-1)
        regrets = []

        for i in range(n_iter):
            # Fit Gaussian process model
            model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1]))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            # Acquisition function (UCB with increased beta for more exploration)
            UCB = UpperConfidenceBound(model, beta=0.2)

            # Optimize acquisition function
            standard_bounds = torch.zeros(2, problem.dim)
            standard_bounds[1] = 1
            candidate, _ = optimize_acqf(UCB, bounds=standard_bounds, q=1, num_restarts=10, raw_samples=50)

            # Evaluate the new candidate
            new_x = unnormalize(candidate, bounds=problem.bounds)
            new_y = problem(new_x).unsqueeze(-1)

            # Update training points
            train_x = torch.cat([train_x, new_x], dim=0)
            train_y = torch.cat([train_y, new_y], dim=0)

            # Compute regret
            best_value_so_far = train_y.min().item()
            regret = best_value_so_far - optimal_value
            regrets.append(regret)

        return train_x, train_y, regrets
    except Exception as e:
        print(f"Bayesian Optimization error: {e}")
        traceback.print_exc()
        return torch.empty((0, 2)), torch.empty((0, 1)), []

# TPE Optimization using Optuna
def tpe_optimization(n_iter=20):
    try:
        def objective(trial):
            x1 = trial.suggest_uniform('x1', -5.0, 10.0)
            x2 = trial.suggest_uniform('x2', -5.0, 10.0)
            x = torch.tensor([[x1, x2]])
            y = problem(x).item()
            return y

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_iter)

        tpe_x = torch.tensor([[trial.params['x1'], trial.params['x2']] for trial in study.trials])
        tpe_y = torch.tensor([[trial.value] for trial in study.trials])
        regrets = []

        for i in range(n_iter):
            best_value_so_far = tpe_y[:i+1].min().item()
            regret = best_value_so_far - optimal_value
            regrets.append(regret)

        return tpe_x, tpe_y, regrets
    except Exception as e:
        print(f"TPE Optimization error: {e}")
        traceback.print_exc()
        return torch.empty((0, 2)), torch.empty((0, 1)), []

# Pareto Front calculation
def compute_pareto_front(y_values):
    # Find non-dominated points for multi-objective optimization
    non_dominated_mask = is_non_dominated(y_values)
    return non_dominated_mask

# Calculate cumulative regret
def calculate_cumulative_regret(regrets):
    return [sum(regrets[:i+1]) for i in range(len(regrets))]

# Plot results including Pareto front
def plot_results(random_x, bayes_x, tpe_x, random_regrets, bayes_regrets, tpe_regrets):
    cumulative_random_regrets = calculate_cumulative_regret(random_regrets)
    cumulative_bayes_regrets = calculate_cumulative_regret(bayes_regrets)
    cumulative_tpe_regrets = calculate_cumulative_regret(tpe_regrets)

    # Plot regret and cumulative regret in the same figure
    fig, axs = plt.subplots(2, 1, figsize=(18, 12))

    # Regret plot
    axs[0].plot(random_regrets, label='Random Sampling')
    axs[0].plot(bayes_regrets, label='Bayesian Optimization')
    axs[0].plot(tpe_regrets, label='TPE Optimization')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Regret')
    axs[0].legend()
    axs[0].set_title('Regret Comparison')

    # Cumulative Regret plot
    axs[1].plot(cumulative_random_regrets, label='Cumulative Random Sampling')
    axs[1].plot(cumulative_bayes_regrets, label='Cumulative Bayesian Optimization')
    axs[1].plot(cumulative_tpe_regrets, label='Cumulative TPE Optimization')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Cumulative Regret')
    axs[1].legend()
    axs[1].set_title('Cumulative Regret Comparison')

    plt.savefig("regret_comparison.png", dpi=300)
    plt.show()

    # Plot contour for Ackley function
    X1_range = torch.linspace(-5, 10, 100)
    X2_range = torch.linspace(-5, 10, 100)
    X1, X2 = torch.meshgrid(X1_range, X2_range)
    X_grid = torch.cat([X1.reshape(-1, 1), X2.reshape(-1, 1)], dim=1)
    Z = problem(X_grid).reshape(X1.shape)

    plt.figure(figsize=(8, 6))
    cp = plt.contour(X1.numpy(), X2.numpy(), Z.numpy(), levels=50, cmap='viridis')
    plt.colorbar(cp, label='Ackley function value')

    # Scatter points for each optimization method
    plt.scatter(random_x[:, 0].numpy(), random_x[:, 1].numpy(), label='Random sampling', color='red', edgecolor='k')
    plt.scatter(bayes_x[:, 0].numpy(), bayes_x[:, 1].numpy(), label='Bayesian Optimization', color='blue', edgecolor='k')
    plt.scatter(tpe_x[:, 0].numpy(), tpe_x[:, 1].numpy(), label='TPE Optimization', color='green', edgecolor='k')

    # Highlight the optimal point
    plt.scatter(optimal_point[0], optimal_point[1], marker='*', color='yellow', s=200, label='Optimal point')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Contour plot of Ackley function with Optimization points')
    plt.savefig("contour_plot.png", dpi=300)
    plt.show()

    # Pareto Front plot
    all_x = torch.cat([random_x, bayes_x, tpe_x], dim=0)
    all_y = torch.cat([problem(random_x), problem(bayes_x), problem(tpe_x)], dim=0)

    pareto_mask = compute_pareto_front(all_y.unsqueeze(-1))
    pareto_points = all_x[pareto_mask]

    plt.figure(figsize=(8, 6))
    plt.scatter(all_x[:, 0].numpy(), all_x[:, 1].numpy(), label='All points', color='gray')
    plt.scatter(pareto_points[:, 0].numpy(), pareto_points[:, 1].numpy(), color='red', label='Pareto Front')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Pareto Front')
    plt.legend()
    plt.savefig("pareto_front.png", dpi=300)
    plt.show()

def main():
    n_iterations = 20

    # Run the optimizations and plot the results
    random_x, random_y, random_regrets = random_sampling(n_iter=n_iterations)
    bayes_x, bayes_y, bayes_regrets = bayesian_optimization(n_iter=n_iterations)
    tpe_x, tpe_y, tpe_regrets = tpe_optimization(n_iter=n_iterations)

    plot_results(random_x, bayes_x, tpe_x, random_regrets, bayes_regrets, tpe_regrets)

if __name__ == "__main__":
    main()
