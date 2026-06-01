# TD Prediction for Estimating the State-Value Function using FrozenLake Environment

## Aim

To implement the Temporal Difference (TD) Prediction algorithm for estimating the state-value function in the FrozenLake environment using Reinforcement Learning.

---

## Algorithm

### TD Prediction Algorithm

1. Import the required libraries.
2. Create the FrozenLake environment using OpenAI Gym.
3. Initialize:
   - Learning rate \( \alpha \)
   - Discount factor \( \gamma \)
   - Number of episodes
   - State-value function \( V(s) \)
4. Define a random policy for action selection.
5. For each episode:
   - Reset the environment.
   - Repeat until the episode ends:
     - Select an action using the policy.
     - Perform the action and observe:
       - Next state
       - Reward
       - Terminal condition
     - Compute TD Target:

\[
TD\ Target = R + \gamma V(S')
\]

     - Compute TD Error:

\[
TD\ Error = TD\ Target - V(S)
\]

     - Update the state-value function:

\[
V(S) = V(S) + \alpha \times TD\ Error
\]

     - Move to the next state.
6. Print the estimated state values.
7. Plot the state-value function using a histogram.

---

## Program

```python

# ============================================================
# TD PREDICTION FOR ESTIMATING STATE-VALUE FUNCTION
# ============================================================

import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Create Environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Parameters
alpha = 0.1
gamma = 0.9
episodes = 5000

# State Value Function
V = defaultdict(float)

# Random Policy
def policy(state):
    return env.action_space.sample()

# TD Prediction Algorithm
for ep in range(episodes):

    state, _ = env.reset()

    done = False

    while not done:

        action = policy(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        # TD Target
        td_target = reward + gamma * V[next_state]

        # TD Error
        td_error = td_target - V[state]

        # Update State Value
        V[state] = V[state] + alpha * td_error

        state = next_state

# Print State Values
print("\nTD State Value Function:\n")

for s in range(env.observation_space.n):
    print(f"State {s}: {V[s]:.4f}")

# ------------------------------------------------
# Histogram Plot
# ------------------------------------------------

states = list(range(env.observation_space.n))
values = [V[s] for s in states]

plt.figure(figsize=(10,5))

plt.bar(states, values)

plt.xlabel("States")
plt.ylabel("Estimated State Value")
plt.title("TD Prediction State Value Function")

plt.show()


```



## Output

<img width="905" height="301" alt="image" src="https://github.com/user-attachments/assets/dbaf845d-6b91-4194-843b-cd695eb91a54" />

## Output Graph
<img width="852" height="423" alt="image" src="https://github.com/user-attachments/assets/6e62add6-33e0-4a98-a1f7-689def6118d8" />

The histogram displays the estimated state-value function for all states in the FrozenLake environment after TD learning.

---

## Result

Thus, the TD Prediction algorithm was successfully implemented in the FrozenLake environment to estimate the state-value function. The agent learned the value of each state through continuous interaction with the environment using Temporal Difference learning.
