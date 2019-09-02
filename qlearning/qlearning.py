import random as rand
import numpy as np
import itertools
import sys
from collections import namedtuple
from datetime import datetime
from lib.utils import *
from os import mkdir

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
        # if count % 1000 == 0:
        #     print(Q[1].values())
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


#####!!!!! vvvvv EPSILON DECAYS vvvvv !!!!!#####

# def q_learning(env, no_episodes, discount_factor=0.9, alpha=0.5, eps_start=1.0, eps_end=0.001, eps_decay=None, sight = float('inf')):
def q_learning(env, no_episodes, discount_factor, alpha, eps_start, eps_end, eps_decay, sight):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    (target policy) while following an epsilon-greedy policy (behaviour policy)

    Args:
        env: OpenAI environment
        no_episodes: Number of episodes to run for
        discount_factor: Lambda time discount factor
        alpha: TD learning rate
        epsilon: Probability of sampling a random action (as ogenerate_graphicspposed to the
        'best' action) with 0 < epsilon < 1

    Returns:
        A tuple (Q, stats)
        Q is the optimal action-value function, a dict of dicts mapping
        state-action pairs to q values
        stats is an EpisodeStats object with two numpy arrays for
        episode_lengths and episode_rewards
    """
    # print('no_episodes',no_episodes, 'discount_factor', discount_factor, 'alpha', alpha, 'epsilon', epsilon, 'sight', sight)

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = initialise_q(env.observation_space.n, env.action_space.n)

    if eps_decay == None:
        eps_decay = (eps_end/eps_start)**(1/no_episodes)
    eps = eps_start

    max_stats_datapoints = 100000
    save_stats_every = int(no_episodes / max_stats_datapoints) + 1 * (no_episodes % max_stats_datapoints != 0)
    stats = namedtuple("Stats",["saved_episodes", "episode_lengths", "episode_rewards"])(
        saved_episodes=[],
        episode_lengths=np.zeros(min(max_stats_datapoints,no_episodes)),
        episode_rewards=np.zeros(min(max_stats_datapoints,no_episodes)))

    running_episode_rewards = 0
    running_episode_lengths = 0
    # flag = 0
    # count = 0

    # The policy we're following
    # behaviour_policy = get_state_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    behaviour_policy = get_state_epsilon_greedy_policy(Q, eps, env.action_space.n)

    for episode in range(1, no_episodes+1):
        # Print out the number of completed episodes in increments of print_interval
        print_interval = 100
        if episode % print_interval == 0:
            print("Episode %i/%i" % (episode, no_episodes))
            sys.stdout.flush()

        # Reset the environment and pick the first action
        true_current_state = env.reset()
        current_state = 
        # count += 1
        for timestep in itertools.count():
            # Take a step
            behaviour_policy = get_state_epsilon_greedy_policy(Q, eps, env.action_space.n)
            action_probs = behaviour_policy(current_state)
            # if episode <= 20:
            #     print(behaviour_policy(1))
            action = np.random.choice(range(len(action_probs)), p=list(action_probs.values()))
            next_state, reward, done, _ = env.step(action)

            # if current_state == 552 and episode%10000 == 0:
            #     print('current_state:', current_state)
            #     print('action:', action)
            #     print('current_Q:', Q[current_state])
            #     print('reward:', reward)
            #     print('next_state:', next_state)
            #     print('next_state_q:', Q[next_state])


            # Update statistics
            # if episode % save_stats_every == 0:
                # stats.episode_rewards[int(episode/save_stats_every)] += reward
            running_episode_rewards += reward


            # TD Update
            max_q_value = max(Q[next_state].values())
            # best_next_actions = [action for action in Q[next_state].keys() if Q[next_state][action] == max_q_value]
            # best_next_action = rand.choice(best_next_actions)
            td_target = reward + discount_factor * max_q_value
            td_delta = td_target - Q[current_state][action]
            Q[current_state][action] += alpha * td_delta

            if done:
                running_episode_lengths += timestep + 1
                if episode % save_stats_every == 0:
                    stats.saved_episodes.append(episode)
                    stats.episode_lengths[int(episode/save_stats_every) - 1] = running_episode_lengths/save_stats_every
                    stats.episode_rewards[int(episode/save_stats_every) - 1] = running_episode_rewards/save_stats_every
                    running_episode_rewards = 0
                    running_episode_lengths = 0
                    # count = 0
                break
            else:
                current_state = next_state
        eps = max(eps_end, eps_decay*eps)
    return Q, stats


def train_q_learning(env, no_episodes, discount_factor=0.99, alpha=0.001, eps_start=1.0, eps_end=0.001, eps_decay=None, sight = float('inf'), save_qvalues = True, save_stats = True):
    start_time = datetime.now().strftime('%Y%m%d_%H%M')

    Q_values, stats = q_learning(env, no_episodes, discount_factor, alpha, eps_start, eps_end, eps_decay, sight)
    # policy = get_greedy_policy(Q_values)

    if save_qvalues:
        qvalues_filename = 'trained_parameters/qlearning_qvalues/' + '_'.join([start_time, str(env.board_height), str(env.board_width), env.reward_type, str(no_episodes), str(discount_factor), str(alpha), str(eps_start), str(eps_end), str(eps_decay), str(sight)]) + '.txt'
        # save_policy_to_file(policy, policy_filename)
        save_qvalues_to_file(Q_values, qvalues_filename)

    if save_stats:
        stats_directory = 'training_analysis/qlearning/' + '_'.join([start_time, str(env.board_height), str(env.board_width), env.reward_type, str(no_episodes), str(discount_factor), str(alpha), str(eps_start), str(eps_end), str(eps_decay), str(sight)])
        mkdir(stats_directory)
        stats_filename = stats_directory + '/stats.txt'
        save_training_analysis_to_file(stats, stats_filename)
        return qvalues_filename, stats_directory

    # return policy_filename, stats_directory, Q_values
    return qvalues_filename, None
