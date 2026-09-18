import numpy as np
from Soft_Actor_Critic import Agent
import gymnasium as gym


if __name__ == '__main__':

    # Create the environment 
    env = gym.make('InvertedPendulum-v4')

    # Create the agent and set parameters
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0], update_temper=True, batch_size=128,reward_scale=0.5, chkpt_dir='tmp/InvertedPendulum-v4/model')
    n_games = 300 # Episodes to train the agent

    # Utilities
    best_score = -np.inf  # Initialize best_score to negative infinity
    score_history = [] # List to keep track of scores

    load_checkpoint = False  # Set to True if you want to load a pre-trained model
    stay_in_middle = False # If True, apply a penalty to the reward if the cart is not in the middle

    # If load_checkoint is True, we want to load and test
    if load_checkpoint:
        env = gym.make('InvertedPendulum-v4', render_mode="human")
        agent.load_models()
        env.reset()
        env.render()

    # Train or test the agent
    for i in range(n_games):
        observation, _ = env.reset()  # Reset the environment and get the initial observation
        done = False # Initialize done to False, which indicates the episode is not finished
        score = 0 # Initialize score to 0, which will accumulate the rewards of the episode

        # Loop through the episode until done (each loop represents one step in the environment)
        while not done:

            # Choose an action based on the current observation
            # If load_checkpoint is True, we want to use the deterministic action
            # If load_checkpoint is False, we want to use the stochastic action
            if load_checkpoint:
                action = agent.choose_action(observation, deterministic=True)
            else:
                action = agent.choose_action(observation)

            # Step the environment with the chosen action
            observation_, reward, terminated, truncated, info = env.step(action)
            
            # Apply a penalty to the reward if the cart is not in the middle
            if stay_in_middle:
                cart_position = observation_[0]  # First element is cart x-position
                position_penalty = 2.0 * (cart_position ** 2)
                reward -= position_penalty  # Apply penalty to the reward

            done = terminated or truncated # Check if the episode is done
            score += reward # Accumulate the reward
            agent.remember(observation, action, reward, observation_, done) # Store the experience in the replay buffer

            # If load_checkpoint is False, we want to learn from the experience
            if not load_checkpoint:
                agent.learn()

            # Update the environment with the new observation for the next step
            observation = observation_

        # Add the score to the score history
        score_history.append(score)
        avg_score = np.mean(score_history[-20:])

        # If model significanlty improved, save the model
        if avg_score > best_score + 50 or i == n_games - 1:
            best_score = avg_score # Update the best score
            if not load_checkpoint:
                agent.save_models() # Save the model

        # Log progress
        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    # Save the metrics and analysis data if not loading a checkpoint
    if not load_checkpoint:

        np.save('tmp/InvertedPendulum-v4/rewards/step_rewards.npy', score_history)
        np.save('tmp/InvertedPendulum-v4/analysis/temperature.npy', agent.temper_history)
        np.save('tmp/InvertedPendulum-v4/analysis/Q1_mean.npy', agent.q1_mean_history)
        np.save('tmp/InvertedPendulum-v4/analysis/Q2_mean.npy', agent.q2_mean_history)
        np.save('tmp/InvertedPendulum-v4/analysis/critic_loss.npy', agent.critic_loss_history)
        np.save('tmp/InvertedPendulum-v4/analysis/actor_loss.npy', agent.actor_loss_history)
        np.save('tmp/InvertedPendulum-v4/analysis/value_loss.npy', agent.value_loss_history)

    # Close the environment
    env.close() 