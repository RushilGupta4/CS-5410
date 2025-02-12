import numpy as np
import matplotlib.pyplot as plt
from math import erf, sqrt, floor, ceil


def build_normal_pmf(mu, sigma, max_demand, cut=4):
    """
    Discretize a Normal(mu, sigma) distribution to form a PMF over {0, 1, ..., max_demand}.
    We use a midpoint approximation and truncate at [mu - cut*sigma, mu + cut*sigma].
    """
    demands = np.arange(0, max_demand + 1)

    def normal_cdf(x):
        return 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))

    pmf = np.zeros_like(demands, dtype=float)
    for i, d in enumerate(demands):
        left = d - 0.5
        right = d + 0.5
        pmf[i] = normal_cdf(right) - normal_cdf(left)
    pmf /= pmf.sum()  # re-normalize (in case of truncation)
    return pmf


def value_iteration(
    M, c, k, h, p, mu, sigma, max_demand, gamma=0.95, max_iter=1000, conv_tol=1e-6
):
    """
    Vectorized (matrix-based) value iteration for the single-product inventory problem.

    Parameters:
      M         : maximum inventory (states are 0,1,...,M)
      c         : cost per unit ordered
      k         : fixed cost incurred if any order is placed (a > 0)
      h         : holding cost per unit of inventory at the beginning of the period
      p         : revenue per unit sold
      mu, sigma : parameters for the (normal) demand distribution
      max_demand: maximum demand (for discretizing the distribution)
      gamma     : discount factor
      max_iter  : maximum number of iterations
      conv_tol  : convergence tolerance

    Returns:
      V      : optimal value function (array of length M+1)
      policy : optimal ordering quantity for each inventory level (array of ints)
    """
    # 1. Build the discrete demand PMF and demand values.
    pmf = build_normal_pmf(mu, sigma, max_demand)  # shape (D,) where D = max_demand+1
    d_vals = np.arange(0, max_demand + 1)  # possible demand values

    # 2. Create grids for states and actions.
    states = np.arange(M + 1)  # states: 0,1,...,M
    actions = np.arange(M + 1)  # allow order quantities 0,1,...,M

    # Create 2-D arrays for state and action combinations.
    X = states.reshape(-1, 1)  # shape (M+1, 1)
    A = actions.reshape(1, -1)  # shape (1, M+1)

    print(A)

    # Total available inventory after ordering:
    inv = X + A  # shape (M+1, M+1)

    # Feasible actions: we cannot order so that x+a exceeds M.
    feasible_mask = inv <= M

    # Fixed cost components (do not depend on demand):
    # - Holding cost on current inventory x.
    # - Ordering cost (variable: c*a and fixed: k if a>0).
    fixed_cost = -h * X - c * A - k * (A > 0)  # shape (M+1, M+1)

    # 3. For each (state, action) pair and each possible demand d,
    # compute:
    #    sales = min(inv, d) and next_state = max(inv - d, 0)
    # Expand dimensions to include demand.
    inv_expanded = inv[:, :, None]  # shape (M+1, M+1, 1)
    d_expanded = d_vals.reshape(1, 1, -1)  # shape (1, 1, D)

    # Compute sales:
    sales = np.minimum(inv_expanded, d_expanded)  # shape (M+1, M+1, D)
    # Compute next state; note the change: we clip the next state to M to avoid indexing errors.
    next_state = np.minimum(np.maximum(inv_expanded - d_expanded, 0), M).astype(
        int
    )  # shape (M+1, M+1, D)

    # 4. Initialize value function V and policy.
    V = np.zeros(M + 1)
    policy = np.zeros(M + 1, dtype=int)

    # 5. Value iteration loop.
    for it in range(max_iter):
        # Compute V_next for every (x,a,d) triple:
        V_next = V[next_state]  # shape (M+1, M+1, D)

        # Total reward for each (state, action, demand):
        total_reward = (
            fixed_cost[:, :, None] + p * sales + gamma * V_next
        )  # shape (M+1, M+1, D)

        # Take expectation over demand:
        expected_Q = np.sum(
            total_reward * pmf.reshape(1, 1, -1), axis=2
        )  # shape (M+1, M+1)

        # Set Q-values for infeasible actions very low.
        expected_Q[~feasible_mask] = -1e15

        # For each state (each row), choose the best action.
        V_new = np.max(expected_Q, axis=1)
        best_actions = np.argmax(expected_Q, axis=1)

        diff = np.linalg.norm(V_new - V)
        V = V_new
        policy = best_actions

        print(f"Iteration {it}, diff = {diff}")
        if diff < conv_tol:
            print(f"Converged at iteration {it} with diff = {diff}")
            break

    return V, policy


if __name__ == "__main__":
    # ------------------------
    # Example parameter setup:
    # ------------------------
    M = 500  # max inventory
    c = 55  # cost per unit
    k = 10  # fixed cost if order >0
    h = 1  # holding cost per unit
    p = 65  # price per unit sold
    mu = 50  # average demand
    sigma = 15  # stdev of demand
    max_demand = 100  # truncate normal at 40
    gamma = 0.99  # discount factor
    max_iter = 20
    conv_tol = 1e-6

    V, policy = value_iteration(
        M, c, k, h, p, mu, sigma, max_demand, gamma, max_iter, conv_tol
    )

    # Print results for a few states
    print("State | Optimal Order | Value")
    n = 10
    for x in [int(i * M / n) for i in range(n + 1)]:
        print(f"{x:5d} | {int(policy[x]):12d} | {V[x]:.4f}")

    # Plot the resulting policy
    plt.figure(figsize=(8, 5))
    plt.bar(range(M + 1), policy, width=1)
    plt.title("Optimal Inventory Policy")
    plt.xlabel("Inventory level (x)")
    plt.ylabel("Order quantity (a)")
    plt.grid(True)
    plt.savefig("inventory_policy.png")
