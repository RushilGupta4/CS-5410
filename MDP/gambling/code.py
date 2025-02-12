import numpy as np
import matplotlib.pyplot as plt

# (S_t: t>= 0) is a sequence of i.i.d samples taking values 1, -1 with probability p, and 1 - p
# We start with wealth X_0, and the gambler can bet A_t \in N, 0 < A_t <= X_t on each round, where A_t is the amount he bets.
# Then, X_{t+1} = X_t + S_t * A_t

# The goal of the gambler is to reach a wealth W*, and will only stop when he reaches that wealth or 0.
# Our goal is to find the optimal policy that maximizes the gambler's wealth (The best A_t for each X_t)


p = 0.1
iters = 100
conv_epsilon = 10 ** (-32)

W_star = 5000
state_steps = 1

STATES = np.arange(0, W_star + state_steps, state_steps)


def value_iteration():
    V = np.zeros(len(STATES))
    V[-1] = 1

    for i in range(iters):
        V_new = V.copy()

        for j, s in enumerate(STATES):
            if s == 0 or s >= W_star:  # Terminal states
                continue

            best_value = 0

            for a in STATES[1:]:
                bet = a
                if bet > s or s + bet > W_star:
                    continue

                win_state = s + bet
                lose_state = s - bet

                win_idx = int(win_state / state_steps)
                lose_idx = int(lose_state / state_steps)

                current_value = p * V[win_idx] + (1 - p) * V[lose_idx]
                best_value = max(current_value, best_value)

            V_new[j] = best_value

        diff = np.linalg.norm(V_new - V)
        V = V_new

        if diff < conv_epsilon:
            print(f"Converged at iteration {i} with diff = {diff}")
            break

        print(f"Iteration {i}, diff = {diff}")

    policy = np.zeros(len(STATES))  # Added policy storage
    for i, s in enumerate(STATES):
        best_value = 0
        best_action = 0

        for a in STATES[1:]:
            bet = a
            if bet > s or s + bet > W_star:
                continue

            win_state = s + bet
            lose_state = s - bet

            win_idx = int(win_state / state_steps)
            lose_idx = int(lose_state / state_steps)

            current_value = p * V[win_idx] + (1 - p) * V[lose_idx]
            if current_value > best_value:
                best_value = current_value
                best_action = a

        policy[i] = best_action

    return V, policy


if __name__ == "__main__":
    V, optimal_policy = value_iteration()

    optimal_policy[1:] = optimal_policy[1:] / STATES[1:]

    # Uncomment if you wish to see a few sample output values:
    # print("Normalized wealth, Optimal bet fraction:")
    # for s, a in zip(
    #     STATES[::1], optimal_policy[::1]
    # ):  # printing every 10th state for brevity
    #     print(f"{s:.2f} -> {a:.3f}")

    # Create a figure with two subplots: the top for the value function, the bottom for the optimal policy.
    fig, axs = plt.subplots(2, 1, figsize=(24, 16))

    # Top subplot: Plot the optimal value function as a line plot.
    axs[0].plot(STATES, V, color="blue", linewidth=2)
    axs[0].set_title("Optimal Value Function")
    axs[0].set_xlabel("Wealth")
    axs[0].set_ylabel("Value")
    axs[0].grid(True)

    # Bottom subplot: Plot the optimal policy as a bar chart.
    axs[1].bar(STATES, optimal_policy, width=state_steps * 0.5, color="green")
    axs[1].set_title("Optimal Policy (Bet Fraction)")
    axs[1].set_xlabel("Wealth")
    axs[1].set_ylabel("Bet Fraction")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("gambling_policy_value.png")
