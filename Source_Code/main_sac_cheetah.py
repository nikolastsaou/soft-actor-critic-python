import numpy as np
from Soft_Actor_Critic import Agent
import gymnasium as gym

if __name__ == '__main__':

    # Create the environment
    # Set the environment to HalfCheetah-v4 with a maximum episode length of 1000 steps
    env = gym.make('HalfCheetah-v4', render_mode="human",max_episode_steps=1000)
    
    # Create the agent and set parameters
    # The agent will use the observation space shape, action space shape, and other parameters
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0], update_temper=True, temper=0.2, batch_size=256,reward_scale=0.5,alpha = 0.0003, beta = 0.001, chkpt_dir='tmp/HalfCheetah-v4/model')
    
    # Number of games (episodes) to train the agent
    n_games = 1000

    # Utilities
    best_score = -np.inf  # Initialize best_score to negative infinity
    score_history = [] # List to keep track of scores
    load_checkpoint = False  # Set to True if you want to load a checkpoint and test the agent

    # If load_checkoint is True, we want to load and test
    if load_checkpoint:
        env = gym.make('HalfCheetah-v4', render_mode="human", max_episode_steps=1000)
        agent.load_models()
        env.reset()
        env.render()

    # Train or test the agent
    for i in range(n_games):
        observation, _ = env.reset()  # Reset the environment and get the initial observation
        done = False # Initialize done to False, which indicates the episode is not finished
        score = 0 # Initialize score to 0, which will accumulate the rewards of the episode

        # Log network losses (average over the last 100 steps)
        print('actor loss: ', np.mean(agent.actor_loss_history[-100:]) if agent.actor_loss_history else 0)
        print('critic loss: ', np.mean(agent.critic_loss_history[-100:]) if agent.critic_loss_history else 0)
        print('value loss: ', np.mean(agent.value_loss_history[-100:]) if agent.value_loss_history else 0)
        
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

            done = terminated or truncated # Check if the episode is done
            score += reward # Accumulate the reward
            agent.remember(observation, action, reward, observation_, done) # Store the experience in the agent's memory

            # If load_checkpoint is False, we want to learn from the experience
            if not load_checkpoint:
                agent.learn()

            # Update the observation for the next step    
            observation = observation_

        score_history.append(score) # Append the score of the episode to the score history
        avg_score = np.mean(score_history[-50:]) # Calculate the average score over the last 50 episodes

        # If model significanlty improved, save the model
        if avg_score > best_score + 50 or i == n_games - 1:
            best_score = avg_score # Update the best score

            if not load_checkpoint:
                agent.save_models() # Save the model if it has significantly improved

        # Print the episode number, score, and average score
        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    # Save metrics for analysis
    if not load_checkpoint:

        np.save('tmp/HalfCheetah-v4/rewards/step_rewards.npy', score_history)
        np.save('tmp/HalfCheetah-v4/analysis/temperature.npy', agent.temper_history)
        np.save('tmp/HalfCheetah-v4/analysis/Q1_mean.npy', agent.q1_mean_history)
        np.save('tmp/HalfCheetah-v4/analysis/Q2_mean.npy', agent.q2_mean_history)
        np.save('tmp/HalfCheetah-v4/analysis/critic_loss.npy', agent.critic_loss_history)
        np.save('tmp/HalfCheetah-v4/analysis/actor_loss.npy', agent.actor_loss_history)
        np.save('tmp/HalfCheetah-v4/analysis/value_loss.npy', agent.value_loss_history)

    # Close the environment    
    env.close() 
