import random as rand
import numpy as np
import itertools
import sys
from collections import namedtuple


def initialise_q(nS, nA):
    # initialises a dict (keyed on state_index) of dicts (keyed on action_index).
    # values are q values of state action pair
    # initially the q value for each action at each state are set to 0
    # Commented out code would initialise q values at random
    Q = {}
    for state_index in range(nS):
        Q[state_index] = {}
        for action_index in range(nA):
            Q[state_index][action_index] = 0
        #     Q[state_index][action_index] = rand.random()
    return Q


def get_greedy_policy(Q):
    policy = {}
    for state_index in range(len(Q)):
        max_q_value = max(Q[state_index].values())
        best_actions = [action for action in Q[state_index].keys() if Q[state_index][action] == max_q_value]
        best_action = rand.choice(best_actions)
        policy[state_index] = best_action
    return policy


def get_state_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dict (keyed on state_index) of dicts(keyed on action_index)
            that maps from state-action pairs to q values
        epsilon: The probability to select a random action with 0 < epsilon < 1
        nA: Number of actions in the environment.

    Returns:
        A function that takes the state as input and returns the probability
        of taking each action as a dict

    """
    def generate_policy(state_index):
        max_q_value = max(Q[state_index].values())
        best_actions = [action for action in Q[state_index].keys() if Q[state_index][action] == max_q_value]
        best_action = rand.choice(best_actions)
        # best_action_index = np.argmax(list(Q[state_index].values()))
        policy = {}
        for action in range(nA):
            if action == best_action:
                policy[action] = 1.0 - epsilon * (nA - 1) / nA
            else:
                policy[action] = epsilon / nA
        return policy

    return generate_policy


def q_learning(env, no_episodes, discount_factor=0.9, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    (target policy) while following an epsilon-greedy policy (behaviour policy)

    Args:
        env: OpenAI environment
        no_episodes: Number of episodes to run for
        discount_factor: Lambda time discount factor
        alpha: TD learning rate
        epsilon: Probability of sampling a random action (as opposed to the
        'best' action) with 0 < epsilon < 1

    Returns:
        A tuple (Q, stats)
        Q is the optimal action-value function, a dict of dicts mapping
        state-action pairs to q values
        stats is an EpisodeStats object with two numpy arrays for
        episode_lengths and episode_rewards
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = initialise_q(env.observation_space.n, env.action_space.n)

    stats = namedtuple("Stats",["episode_lengths", "episode_rewards"])(
        episode_lengths=np.zeros(no_episodes),
        episode_rewards=np.zeros(no_episodes))

    # The policy we're following
    behaviour_policy = get_state_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for episode in range(1, no_episodes+1):
        # Print out the number of completed episodes in increments of print_interval
        print_interval = 50000
        if (episode) % print_interval == 0:
            print("Episode %i/%i" % (episode, no_episodes+1))
            sys.stdout.flush()

        # Reset the environment and pick the first action
        current_state = env.reset()

        for timestep in itertools.count():
            # Take a step
            action_probs = behaviour_policy(current_state)
            action = np.random.choice(range(len(action_probs)), p=list(action_probs.values()))
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[episode-1] += reward
            stats.episode_lengths[episode-1] = timestep

            # TD Update
            max_q_value = max(Q[next_state].values())
            best_next_actions = [action for action in Q[next_state].keys() if Q[next_state][action] == max_q_value]
            best_next_action = rand.choice(best_next_actions)
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[current_state][action]
            Q[current_state][action] += alpha * td_delta

            if done:
                break

            current_state = next_state

    return Q, stats
