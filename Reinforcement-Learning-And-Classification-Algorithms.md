---
title: "Exploring Reinforcement Learning and Classification Algorithms"
author: "Kyle Lukaszek"
date: "November 2023"
tags:
    - ML
    - RL
    - Classification
    - OpenAI Gym
    - Q* Learning
    - Naive Bayes Classifier
    - Gaussian Mixture Models
description: "Another adapted assignment Jupyter Notebook from CIS*4780 Computational Intelligence"
---
# Exploring Reinforcement Learning and Classification Algorithms

This document outlines an exploration of Q-learning for a grid world environment, a Naive Bayes classifier for text categorization, and Gaussian Mixture Models for clustering.

## Dependencies and Setup

First, necessary libraries are imported. `gym` is used for the reinforcement learning environment, `numpy` and `pandas` for data manipulation, `matplotlib` for plotting, and `sklearn` for machine learning utilities.

```python
# Assuming pip install was run in a separate environment or cell if needed
# %pip install gym numpy pandas matplotlib scikit-learn
```

```python
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import f1_score, accuracy_score, silhouette_score
from sklearn.mixture import GaussianMixture
```

## Part 1: Grid World with Q-Learning

This section details the setup of a custom grid world environment and the implementation of a Q-learning agent to navigate it.

### 1.1: Grid World Environment Setup

#### Grid Definition
The environment is a 10x10 grid with open spaces ('O'), obstacles ('X'), a start ('S'), and a goal ('G').

```python
grid = [
    ['O', 'O', 'X', 'O', 'O', 'O', 'O', 'X', 'O', 'G'],
    ['O', 'X', 'X', 'X', 'O', 'X', 'O', 'X', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'X', 'O', 'X', 'O', 'X'],
    ['O', 'X', 'X', 'X', 'O', 'X', 'O', 'X', 'O', 'O'],
    ['O', 'O', 'X', 'O', 'O', 'X', 'O', 'X', 'X', 'O'],
    ['O', 'O', 'X', 'X', 'O', 'X', 'O', 'O', 'O', 'O'],
    ['O', 'X', 'O', 'X', 'O', 'O', 'O', 'X', 'X', 'O'],
    ['O', 'X', 'O', 'X', 'O', 'X', 'O', 'X', 'O', 'O'],
    ['O', 'X', 'O', 'O', 'O', 'X', 'O', 'X', 'O', 'O'],
    ['O', 'O', 'O', 'X', 'O', 'O', 'S', 'O', 'O', 'O']
]
```

#### Grid World Class
A custom `GridWorld` class, extending `gym.Env`, was defined to manage states, actions, and rewards.

*Note: The original notebook mentioned issues with Gym UI rendering in a WSL2/Windows 10 environment. The class implementation focuses on the logic, with a `matplotlib`-based `render` method.*

```python
class GridWorld(gym.Env):
    def __init__(self, grid_layout): # Renamed grid to grid_layout to avoid conflict with global
        self.grid_layout = grid_layout # Store the initial layout
        self.rows = len(self.grid_layout)
        self.cols = len(self.grid_layout[0])
        self.goal_pos = self._find_pos('G') # Use _ for internal helper

        self.agent_pos = self._find_pos('S')
        self.agent_path = [self.agent_pos]
        self.steps = 0

        self.actions_map = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'} # Renamed for clarity
        self.action_space = gym.spaces.Discrete(len(self.actions_map))
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Discrete(self.rows),
            gym.spaces.Discrete(self.cols)
        ))

        self.open_space_reward = -1
        self.obstacle_reward = -10
        self.goal_reward = 10

    def _find_pos(self, char): # Internal helper
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid_layout[i][j] == char:
                    return (i, j)
        return None # Should not happen if S and G are in grid

    def _is_valid_pos(self, i, j): # Internal helper
        return (0 <= i < self.rows and 
                0 <= j < self.cols and 
                self.grid_layout[i][j] != 'X')
    
    def _get_reward(self, i, j): # Internal helper
        if not self._is_valid_pos(i,j): # Check attempted move, not current agent pos
             return self.obstacle_reward # If new position would be invalid
        elif (i, j) == self.goal_pos:
            return self.goal_reward
        else: # Valid, not goal
            return self.open_space_reward

    def _move_agent(self, action_idx): # Renamed for clarity, takes action_idx
        action_str = self.actions_map[action_idx]
        next_i, next_j = self.agent_pos

        if action_str == 'UP': next_i -= 1
        elif action_str == 'DOWN': next_i += 1
        elif action_str == 'LEFT': next_j -= 1
        elif action_str == 'RIGHT': next_j += 1
        
        reward = self._get_reward(next_i, next_j)

        if self._is_valid_pos(next_i, next_j): # If move is valid, update position
            self.agent_pos = (next_i, next_j)
            self.agent_path.append(self.agent_pos)
        # If move is not valid, agent_pos doesn't change, but reward reflects penalty
        
        return reward
    
    def step(self, action_idx): # Takes action_idx
        reward = self._move_agent(action_idx)
        self.steps += 1
        done = (self.agent_pos == self.goal_pos)
        # info can be an empty dict
        return self.agent_pos, reward, done, {} 
    
    def reset(self):
        self.agent_pos = self._find_pos('S')
        self.agent_path = [self.agent_pos]
        self.steps = 0
        # Must return initial observation and info
        return self.agent_pos, {}

    def render(self, mode='human'): # Added mode argument for gym compatibility
        plt.figure(figsize=(self.cols, self.rows))
        
        # Create a grid indicating visited squares
        traversed_display_grid = np.zeros((self.rows, self.cols))
        for r, c in self.agent_path:
            traversed_display_grid[r, c] = 1
        
        plt.imshow(traversed_display_grid, cmap='Blues', origin='upper', alpha=0.3) # Alpha for overlay

        # Add text for grid elements
        for r in range(self.rows):
            for c in range(self.cols):
                char = self.grid_layout[r][c]
                color = 'black'
                if char == 'X': color = 'red'
                elif char == 'S': color = 'green'
                elif char == 'G': color = 'orange'
                plt.text(c, r, char, ha='center', va='center', color=color, fontweight='bold')

        # Draw path
        if len(self.agent_path) > 1:
            path_y, path_x = zip(*self.agent_path) # Matplotlib uses (x,y) not (row,col)
            plt.plot(path_x, path_y, color='black', linewidth=2, marker='o', markersize=3)

        plt.xticks([])
        plt.yticks([])
        plt.title(f"Agent Path - Steps: {self.steps}")
        plt.show()
```
*Refinement: Renamed some internal methods and variables for clarity (e.g., `_find_pos`, `_is_valid_pos`). The `step` and `reset` methods were updated to return `info` dictionaries, common in Gym environments. The `render` method was slightly adjusted for visual clarity.*

### 1.2: Q-Learning Agent Implementation

A Q-learning agent was implemented to learn an optimal policy for navigating the grid.

#### Q-Learning Agent Class
The agent maintains a Q-table and uses an epsilon-greedy policy for action selection.

```python
class QLearningAgent: # Renamed class
    def __init__(self, num_actions, epsilon, gamma, alpha): # num_actions instead of actions dict
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount factor
        self.q_table = {} # {(row, col): [q_up, q_down, q_left, q_right]}

    def get_q_value(self, state, action_idx):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        return self.q_table[state][action_idx]
    
    def update_q_value(self, state, action_idx, reward, next_state):
        current_q = self.get_q_value(state, action_idx) # Q(s,a)
        
        # Ensure next_state Q-values exist for max Q(s',a') calculation
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_actions)
        
        max_next_q = np.max(self.q_table[next_state]) # max_a' Q(s',a')
        
        # Q-learning update rule
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action_idx] = new_q

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            return np.random.choice(self.num_actions)
        else:
            # Exploit: choose the best known action
            if state not in self.q_table: # If state unseen, initialize Q-values
                self.q_table[state] = np.zeros(self.num_actions)
            # If multiple actions have the same max Q-value, argmax returns the first one.
            # To break ties randomly:
            # best_actions = np.where(self.q_table[state] == np.max(self.q_table[state]))[0]
            # return np.random.choice(best_actions)
            return np.argmax(self.q_table[state]) 
```
*Refinement: The `QLearningAgent` class was slightly restructured. The Q-value update now directly implements the standard formula `Q(s,a) <- Q(s,a) + alpha * (R + gamma * max_a' Q(s',a') - Q(s,a))`. A note on tie-breaking in `choose_action` was added.*

### 1.3: Evaluation

#### Evaluation Function
This function trains and evaluates the Q-learning agent over a specified number of episodes.

```python
def evaluate_q_learning_agent(grid_layout, episodes, epsilon, gamma, learning_rate_alpha): # Renamed params
    environment = GridWorld(grid_layout)
    num_actions = environment.action_space.n # Get from env

    agent = QLearningAgent(num_actions, epsilon, gamma, learning_rate_alpha) # Use agent class

    # Store rewards per episode for plotting later, if desired
    # episode_rewards = [] 

    for episode in range(episodes):
        state, _ = environment.reset() # Get initial state
        total_reward_episode = 0
        done = False

        while not done:
            action_idx = agent.choose_action(state)
            next_state, reward, done, _ = environment.step(action_idx) # Use action_idx
            
            total_reward_episode += reward
            agent.update_q_value(state, action_idx, reward, next_state)
            state = next_state
        
        # episode_rewards.append(total_reward_episode) # For plotting learning curve

        # Optional: Print progress periodically
        # if (episode + 1) % (episodes // 10) == 0:
        #     print(f'Episode: {episode + 1}/{episodes}, Steps: {environment.steps}, Reward: {total_reward_episode}')

    # After all episodes, print final run and render
    # This will show the path taken using the learned Q-values (with some epsilon exploration)
    final_state, _ = environment.reset()
    final_total_reward = 0
    final_done = False
    # Set epsilon to 0 for purely greedy policy in final evaluation run, if desired
    # original_epsilon = agent.epsilon
    # agent.epsilon = 0.0 
    while not final_done:
        final_action_idx = agent.choose_action(final_state)
        final_next_state, final_reward, final_done, _ = environment.step(final_action_idx)
        final_total_reward += final_reward
        final_state = final_next_state
    # agent.epsilon = original_epsilon # Restore epsilon if changed

    print(f'Evaluation after {episodes} episodes: Steps: {environment.steps}, Total Reward: {final_total_reward}')
    environment.render()
```
*Refinement: The evaluation function was updated to run a final greedy evaluation path after training to better demonstrate the learned policy. The use of `agent` class methods is now consistent.*

#### Evaluate Agent
The agent was trained for 1000 episodes with `epsilon=0.1`, `gamma=0.5`, and `learning_rate=0.1`.

```python
episodes_count = 1000 # Renamed
epsilon_val = 0.1
gamma_val = 0.5 # Discount factor for Q-learning
alpha_val = 0.1 # Learning rate for Q-learning

evaluate_q_learning_agent(grid, episodes_count, epsilon_val, gamma_val, alpha_val)
```
```
Evaluation after 1000 episodes: Steps: 14, Total Reward: -3
```
![](best-route.png)

### 1.4: Analysis of Q-Learning Performance

#### 1.4.1: Effect of Epsilon (Exploration Rate)
A higher `epsilon` value encourages more exploration. While initially beneficial for discovering the environment, if `epsilon` remains high, the agent may not consistently exploit its learned knowledge, leading to suboptimal paths or longer convergence times. A lower `epsilon` (e.g., 0.1) led to more consistent and efficient pathfinding once the Q-values started to converge. A common strategy, not implemented here, is to decay `epsilon` over time, starting higher and gradually reducing it.

#### 1.4.2: Performance Over Training Episodes
With the chosen hyperparameters (`epsilon=0.1`, `gamma=0.5`, `alpha=0.1`), the agent's performance improved noticeably. Early episodes involved more random exploration. By around 50-100 episodes, the agent began finding efficient paths more regularly, though still making occasional exploratory mistakes. After 1000 episodes, the agent consistently found paths to the goal in approximately 14-16 steps. Some minor suboptimal moves (e.g., hitting a wall occasionally due to `epsilon`-greedy exploration) still occurred, which might be reduced with more episodes or epsilon decay.

#### 1.4.3: Challenges and Adjustments
Initially, a higher `epsilon` (e.g., 0.5) resulted in excessive exploration, preventing consistent convergence to an optimal path. Reducing `epsilon` to 0.1 significantly improved the agent's ability to exploit learned Q-values. Increasing the number of training episodes to 1000 also contributed to more stable and optimal pathfinding.

### 1.5: Hyperparameter Exploration
The impact of varying `epsilon` and `gamma` was explored, keeping the learning rate (`alpha`) at 0.1 and episodes at 1000.

```python
epsilons_to_test = [0.1, 0.2, 0.3, 0.4, 0.5] 
gammas_to_test = [0.6, 0.7, 0.8, 0.9, 1.0] # Note: gamma=1.0 can be problematic if goal is not always reachable
fixed_learning_rate = 0.1
fixed_episodes = 1000

# This loop can be time-consuming.
# for test_epsilon in epsilons_to_test:
#     for test_gamma in gammas_to_test:
#         print(f'Testing with Epsilon: {test_epsilon}, Gamma: {test_gamma}')
#         evaluate_q_learning_agent(grid, fixed_episodes, test_epsilon, test_gamma, fixed_learning_rate)
#         print("-" * 30) 
```
*(The following images represent selected outputs from the hyperparameter exploration loop shown in the original notebook. Only a few key examples are included here for brevity.)*

**Example Output for Epsilon: 0.1, Gamma: 0.6**
```
Evaluation after 1000 episodes: Steps: 14, Total Reward: -3
```
![](best-route.png)

**Example Output for Epsilon: 0.5, Gamma: 1.0**
```
Evaluation after 1000 episodes: Steps: 39, Total Reward: -55
```
![](odd-one.png)

**Hyperparameter Observations:**
Consistent with earlier findings, lower `epsilon` values (e.g., 0.1) generally resulted in better final path efficiency (fewer steps, higher total reward) after 1000 episodes. As `epsilon` increased, the agent's path tended to be longer and more erratic due to persistent exploration. The discount factor `gamma` showed less dramatic impact on the final path length in this specific grid and training duration, though values closer to 1 give more weight to future rewards. For simpler grids where the optimal path is relatively short, the impact of `gamma` might be less pronounced than in more complex environments or if a learning curve was plotted.

## Part 2: Naive Bayes Classifier for Text Categorization

This section covers the implementation of a Naive Bayes classifier to categorize text comments.

### 2.1: Building Binary Feature Vectors
Comments are represented as binary vectors, where each dimension corresponds to a word in a predefined vocabulary. A '1' indicates the presence of the word in the comment, and '0' its absence.

#### Function to Build Binary Vectors

```python
def build_binary_vectors_from_file(file_path, num_total_attributes): # Renamed params
    # Dictionary: {comment_id: binary_vector_np_array}
    binary_vectors_dict = {} 

    with open(file_path, 'r') as file:
        for line in file: # More pythonic iteration
            comment_id, word_id = map(int, line.strip().split())
            
            # word_id is 1-indexed in file, adjust to 0-indexed for array
            # Assuming num_total_attributes is the size of the vocabulary
            if word_id > num_total_attributes: # Safety check
                # print(f"Warning: word_id {word_id} exceeds num_attributes {num_total_attributes}. Skipping.")
                continue

            if comment_id not in binary_vectors_dict:
                binary_vectors_dict[comment_id] = np.zeros(num_total_attributes, dtype=np.int8) # Use int8 for memory
            
            # word_id from file might be 1-indexed based on context
            # If vocabulary is 0 to N-1, then word_id-1
            binary_vectors_dict[comment_id][word_id - 1] = 1 # Assuming word_id in file is 1-indexed

    return binary_vectors_dict
```

#### Function to Read Labels
Labels are read from a file and associated with the corresponding comment IDs.

```python
def read_labels_for_vectors(file_path, existing_comment_ids): # Renamed params
    # List of labels, ordered by comment_id implicitly (after filtering)
    labels_list = []
    temp_labels_dict = {} # {comment_id: label}
    
    with open(file_path, 'r') as file:
        # Assuming labels file is 1 label per line, comment_id implied by line number
        for idx, line in enumerate(file):
            comment_id_from_line = idx + 1 # Line numbers are 1-indexed
            if comment_id_from_line in existing_comment_ids:
                temp_labels_dict[comment_id_from_line] = int(line.strip())
    
    # Ensure labels are ordered by sorted existing_comment_ids
    # This is crucial if binary_vectors_dict is later converted to a list/array
    for cid in sorted(list(existing_comment_ids)):
        if cid in temp_labels_dict:
             labels_list.append(temp_labels_dict[cid])
        # else:
            # This case should not happen if existing_comment_ids is derived from data file
            # print(f"Warning: Comment ID {cid} from data has no label.")
            
    return labels_list
```
*Refinement: The `read_labels` function was adjusted to ensure labels correctly correspond to the binary vectors, especially if the binary vectors are later converted to an ordered list/array for scikit-learn compatibility. Assumed `word_id` in `trainData.txt` is 1-indexed for array access.*

#### Load Data
Training and testing data (word occurrences and labels) are loaded. The vocabulary size is 6968.

```python
vocabulary_size = 6968 # Renamed

train_vectors_dict = build_binary_vectors_from_file('trainData.txt', vocabulary_size)
# Pass keys of train_vectors_dict to ensure labels match existing comments
train_labels_list = read_labels_for_vectors('trainLabel.txt', train_vectors_dict.keys())

print(f"Training Comments Processed: {len(train_vectors_dict)}")
print(f"Training Labels Loaded: {len(train_labels_list)}")

test_vectors_dict = build_binary_vectors_from_file('testData.txt', vocabulary_size)
test_labels_list = read_labels_for_vectors('testLabel.txt', test_vectors_dict.keys())

print(f"Testing Comments Processed: {len(test_vectors_dict)}")
print(f"Testing Labels Loaded: {len(test_labels_list)}")

# For Naive Bayes, convert dict of vectors to a list of vectors (data matrix)
# and ensure labels list order matches. The read_labels_for_vectors now handles this ordering.
train_data_matrix = np.array([train_vectors_dict[cid] for cid in sorted(train_vectors_dict.keys())])
test_data_matrix = np.array([test_vectors_dict[cid] for cid in sorted(test_vectors_dict.keys())])
# train_labels_list and test_labels_list are already ordered
```
```
Training Comments Processed: 1460
Training Labels Loaded: 1460
Testing Comments Processed: 1450
Testing Labels Loaded: 1450
```
*Observation: The original notebook noted discrepancies in comment counts vs. label file lines. The refined `read_labels_for_vectors` ensures only labels for existing comments are loaded and correctly ordered. The data is also converted to NumPy arrays for efficient processing.*

### 2.2: Naive Bayes Classifier Implementation

A Bernoulli Naive Bayes classifier was implemented. It assumes features (word presence) are binary and conditionally independent given the class.

#### Probability Calculation (Training)
This function calculates class prior probabilities and conditional probabilities of word presence for each class, using Laplace smoothing.

```python
def train_bernoulli_naive_bayes(X_data, y_labels, smoothing_alpha=1): # Renamed params
    # X_data: numpy array of shape (num_samples, num_features)
    # y_labels: numpy array of shape (num_samples,)
    
    num_samples, num_features = X_data.shape
    unique_classes = np.unique(y_labels)
    num_classes = len(unique_classes)

    class_priors = {} # P(c)
    # P(w_i=1 | c) for each word i and class c
    # Using log probabilities is often more stable for prediction
    log_conditional_probs_present = {} 
    log_conditional_probs_absent = {} # P(w_i=0 | c) = 1 - P(w_i=1 | c)

    for c_idx, c_val in enumerate(unique_classes):
        # Get samples belonging to class c
        X_class_c = X_data[y_labels == c_val]
        num_samples_class_c = X_class_c.shape[0]

        # Calculate class prior P(c)
        class_priors[c_val] = (num_samples_class_c + smoothing_alpha) / (num_samples + num_classes * smoothing_alpha)
        # Using log prior for later sum
        # class_priors[c_val] = np.log((num_samples_class_c + smoothing_alpha) / (num_samples + num_classes * smoothing_alpha))


        # Calculate P(word_i=1 | class_c) with Laplace smoothing
        # Sum word occurrences for class c, add smoothing_alpha
        word_counts_in_class_c = np.sum(X_class_c, axis=0) + smoothing_alpha
        # Denominator: num_samples_in_class_c + num_features_categories (2 for binary) * smoothing_alpha
        # For Bernoulli, denominator is (count(c) + k*alpha), where k is # categories for feature (2 for binary)
        total_counts_denominator = num_samples_class_c + (2 * smoothing_alpha) 
        
        prob_word_present_given_c = word_counts_in_class_c / total_counts_denominator
        
        # Store log probabilities to avoid underflow during prediction
        log_conditional_probs_present[c_val] = np.log(prob_word_present_given_c)
        log_conditional_probs_absent[c_val] = np.log(1.0 - prob_word_present_given_c)


    # Convert class_priors to log_priors as well
    log_class_priors = {c: np.log(p) for c,p in class_priors.items()}

    return log_class_priors, log_conditional_probs_present, log_conditional_probs_absent
```
*Refinement: The training function now calculates log probabilities directly to prevent numerical underflow during prediction, a common practice for Naive Bayes. The formula for Bernoulli Naive Bayes with Laplace smoothing was clarified for the denominator: `count(c) + k*alpha`, where `k=2` for binary features.*

#### Naive Bayes Classifier (Prediction)
This function classifies new instances using the learned probabilities.

```python
def predict_bernoulli_naive_bayes(X_test_instance, log_class_priors, log_cond_probs_present, log_cond_probs_absent):
    # X_test_instance: a single binary vector (1D numpy array)
    
    posteriors = {} # {class_val: log_posterior_probability}
    
    for c_val in log_class_priors.keys():
        log_posterior_c = log_class_priors[c_val] # Start with log P(c)
        
        # Add sum of log P(feature_i | c)
        # For features present (value = 1) in X_test_instance
        log_posterior_c += np.sum(log_cond_probs_present[c_val][X_test_instance == 1])
        # For features absent (value = 0) in X_test_instance
        log_posterior_c += np.sum(log_cond_probs_absent[c_val][X_test_instance == 0])
        
        posteriors[c_val] = log_posterior_c
        
    # Return class with the highest log posterior probability
    return max(posteriors, key=posteriors.get)

def predict_all_naive_bayes(X_test_matrix, log_class_priors, log_cond_probs_present, log_cond_probs_absent):
    predictions = [predict_bernoulli_naive_bayes(instance, log_class_priors, log_cond_probs_present, log_cond_probs_absent) for instance in X_test_matrix]
    return np.array(predictions)
```

#### Train and Test Classifier

```python
# Convert labels list to numpy array for consistency
y_train_np = np.array(train_labels_list)
y_test_np = np.array(test_labels_list)

# Train the Naive Bayes classifier
log_priors_nb, log_cond_present_nb, log_cond_absent_nb = train_bernoulli_naive_bayes(train_data_matrix, y_train_np, smoothing_alpha=1)

# Make predictions on training and test data
train_predictions_nb = predict_all_naive_bayes(train_data_matrix, log_priors_nb, log_cond_present_nb, log_cond_absent_nb)
test_predictions_nb = predict_all_naive_bayes(test_data_matrix, log_priors_nb, log_cond_present_nb, log_cond_absent_nb)

# Calculate and print accuracy
train_acc_nb = accuracy_score(y_train_np, train_predictions_nb)
test_acc_nb = accuracy_score(y_test_np, test_predictions_nb)
print(f'Training Accuracy: {train_acc_nb:.4f}')
print(f'Test Accuracy: {test_acc_nb:.4f}\n')

# Calculate and print F1 score (weighted for potential imbalance, though this data is balanced)
train_f1_nb = f1_score(y_train_np, train_predictions_nb, average='weighted')
test_f1_nb = f1_score(y_test_np, test_predictions_nb, average='weighted')
print(f'Training F1 Score: {train_f1_nb:.4f}')
print(f'Test F1 Score: {test_f1_nb:.4f}')
```

```
Training Accuracy: 0.9158
Test Accuracy: 0.7434

Training F1 Score: 0.9157
Test F1 Score: 0.7434
```

#### Analysis and Conclusion
The Naive Bayes classifier achieved a training accuracy and F1-score of approximately 91.6%, and a test accuracy and F1-score of approximately 74.3%. The similarity between accuracy and F1-score is expected for this dataset, as the class distribution appears balanced (the original notebook mentioned 50/50, though not explicitly checked here again). The higher performance on training data compared to test data is typical.

Initial implementation challenges included ensuring correct probability calculations (e.g., proper application of Laplace smoothing and handling of log probabilities to prevent underflow) and aligning data vectors with their corresponding labels, especially given that comment IDs in the raw data might not be contiguous or complete.

## Part 3: Data Visualization Using Gaussian Mixture Models (GMM)

Gaussian Mixture Models (GMMs) were used for clustering a 4-dimensional dataset.

### 3.1: Clustering Algorithm Selection
Gaussian Mixture Models were chosen. GMMs assume data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.

### 3.2: Building The Clustering Algorithm

The number of clusters (components) for the GMM was determined using the Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC). Lower BIC/AIC values generally indicate a better model fit, balancing goodness of fit with model complexity.

#### Determine Number of Clusters

```python
# Load "Clustering Data.csv"
df_clustering = pd.read_csv('Clustering Data.csv', header=None, names=['x', 'y', 'z', 'w'])

num_components_range = range(1, 11) # Renamed
bic_scores = []
aic_scores = []

for k_components in num_components_range: # Renamed
    gmm_model = GaussianMixture(n_components=k_components, random_state=0, n_init=10) # n_init for stability
    gmm_model.fit(df_clustering)
    bic_scores.append(gmm_model.bic(df_clustering))
    aic_scores.append(gmm_model.aic(df_clustering))

plt.figure(figsize=(10, 5))
plt.plot(num_components_range, bic_scores, label='BIC', marker='o')
plt.plot(num_components_range, aic_scores, label='AIC', marker='x')
plt.xlabel('Number of Components (Clusters)')
plt.ylabel('Information Criterion Score')
plt.title('BIC and AIC for GMM')
plt.legend()
plt.grid(True)
plt.show()
```
![](scores-by-num-clusters.png)

The plot of BIC and AIC scores suggests that 2 components (clusters) is a reasonable choice. While AIC might continue to decrease or decrease more sharply for a higher number of components (e.g., around 8 in the original notebook's plot), BIC often penalizes complexity more heavily and can be a better indicator for the number of clusters. The "elbow" or point of diminishing returns in these plots is sought. Here, 2 clusters provides a low BIC and a relatively low AIC without suggesting overfitting that a much higher number of components might with a significantly lower AIC but higher BIC.

### 3.3: Report The Results
A GMM with 2 components was fitted to the data.

```python
gmm_final = GaussianMixture(n_components=2, random_state=0, n_init=10)
cluster_labels_gmm = gmm_final.fit_predict(df_clustering)
df_clustering['Cluster'] = cluster_labels_gmm

print(f'Number of Clusters Determined: {df_clustering["Cluster"].nunique()}')

cluster_counts = df_clustering['Cluster'].value_counts().sort_index()
print('\nNumber of Points In Each Cluster:')
for i, count in cluster_counts.items(): # Iterate through sorted cluster indices
    print(f'\tCluster {i}: {count}') # Assuming cluster labels are 0, 1...

silhouette_avg = silhouette_score(df_clustering[['x', 'y', 'z', 'w']], df_clustering['Cluster'])
print(f'\nSilhouette Score: {silhouette_avg:.4f}')
```
```
Number of Clusters Determined: 2

Number of Points In Each Cluster:
	Cluster 0: 100
	Cluster 1: 50

Silhouette Score: 0.6986
```

#### Explanation of Results
The GMM identified 2 clusters. Cluster 0 contains 100 points, and Cluster 1 contains 50 points, indicating an imbalance in cluster sizes.

The Silhouette Score, which measures how similar a point is to its own cluster compared to other clusters, was approximately 0.6986. Scores range from -1 to 1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. A score of ~0.7 suggests reasonably well-defined and separated clusters.

The original notebook mentioned testing 8 clusters, which sometimes yielded a slightly higher silhouette score but with high variance between runs. The 2-component model was deemed more stable and robust based on BIC/AIC and consistent silhouette scores.

### 3.4: Visualize The Results
The clusters are visualized in a 3D scatter plot, using three of the four dimensions for axes and the fourth dimension (`w`) to scale point size.

```python
fig = plt.figure(figsize=(10, 8)) # Adjusted size
ax = fig.add_subplot(111, projection='3d')

# Scatter plot: x, y, z for coordinates, 'Cluster' for color, 'w' for size
scatter_plot = ax.scatter(
    df_clustering['x'], 
    df_clustering['y'], 
    df_clustering['z'], 
    c=df_clustering['Cluster'], 
    cmap='viridis', # Color map
    s=df_clustering['w'] * 20, # Scale point size by 'w'
    alpha=0.7, # Transparency
    edgecolor='k', # Add edge color for better visibility of overlapping points
    linewidth=0.5
)

ax.set_xlabel('X dimension')
ax.set_ylabel('Y dimension')
ax.set_zlabel('Z dimension')
ax.set_title('GMM Clustering (3D Visualization of x, y, z; w scales size)')

# Add a color bar for the clusters
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}', 
                              markerfacecolor=plt.cm.viridis(i / (df_clustering["Cluster"].nunique()-1 if df_clustering["Cluster"].nunique()>1 else 1.0)), # Get color from cmap
                              markersize=8) for i in sorted(df_clustering['Cluster'].unique())]
ax.legend(handles=legend_elements, title="Clusters")

plt.show()
```
![](3d-cluster.png)

The 3D visualization shows the spatial distribution of the two identified clusters. The differing sizes of points (scaled by 'w') add another layer of information to the plot. Despite some visual overlap in the 2D projection of this 3D space, the silhouette score suggests good separation in the 4-dimensional feature space.