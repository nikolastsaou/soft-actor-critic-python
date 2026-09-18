import numpy as np
from Soft_Actor_Critic import Agent
import gymnasium as gym

if __name__ == '__main__':
    
    # Create the environment
    # Set the environment to Hopper-v5 with a maximum episode length of 1000 steps
    env = gym.make('Hopper-v5', render_mode="human",max_episode_steps=1000)

    # Create the agent and set parameters
    # The agent will use the observation space shape of the environment, the number of actions, and other hyperparameters
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0], update_temper=False, batch_size=256, reward_scale=1.0, alpha = 0.0003, beta = 0.0003, chkpt_dir=f'tmp/Hopper-v5/model')

    # Total number of steps
    n_steps = 1000000 # Total number of steps to train the agent (1 million steps)
    LEARN_START = 10000  # Number of steps before the agent starts learning



    # Utilities
    best_score = -np.inf  # Initialize best_score to negative infinity
    score_history = [] # List to keep track of scores

    step_count = 0 # Total number of steps taken by the agent
    episode_count = 0 # Total number of episodes completed
    
    # List to keep track of total steps at the end of each episode
    # This will be used for analysis and plotting later
    episode_steps = [] 

    load_checkpoint = False # Set to True if you want to loas a checkpoint and test the agent
    
    # If load_checkpoint is True, we want to load and test
    if load_checkpoint:
        env = gym.make('Hopper-v5', render_mode="human", max_episode_steps=1000)
        agent.load_models()
        env.reset()
        env.render()

    # Train or test the agent
    while step_count < n_steps:
        observation, _ = env.reset() # Reset the environment and get the initial observation
        done = False # Initialize done to False, which indicates whether the episode is finished
        score = 0 # Initialize score to 0, which will accumulate rewards during the episode
        episode_step_count = 0  # Τrack steps within current episode

        episode_count += 1 # Increment the episode count
        
        while not done:
            # If step_count reaches 1 million, break the loop
            if step_count == 1000000:
                break

            # Choose an action based on the current observation
            # If load_checkpoint is True, choose action deterministically
            # Otherwise, choose action based on the agent's policy
            if load_checkpoint:
                action = agent.choose_action(observation, deterministic=True)
            else:
                action = agent.choose_action(observation)

            # Take a step in the environment with the chosen action
            next_obs, reward, terminated, truncated, info = env.step(action)
        
            step_count += 1 # Increment total step counter
            episode_step_count += 1  # Increment episode step counter

            done = terminated or truncated # Check if the episode is done
            score += reward # Accumulate the reward

            agent.remember(observation, action, reward, next_obs, done) # Store the transition in the agent's memory

            # If the agent has enough experience, it will start learning
            if not load_checkpoint and step_count > LEARN_START:
                agent.learn()

            # Update the observation to the next observation
            observation = next_obs

        
        episode_steps.append(step_count) # Track the total steps at the end of the episode
        score_history.append(score) # Append the score of the episode to the score history
        avg_score = np.mean(score_history[-50:]) # Calculate the average score of the last 50 episodes

        # If model significantly improved, save the model
        if avg_score > best_score + 50:
            best_score = avg_score # Update the best score

            if not load_checkpoint:
                agent.save_models() # Save the model if the average score improved significantly

        # Print episode count, score, average score, total steps, and steps in the current episode
        print(f'episode {episode_count} score {score:.1f} avg_score {avg_score:.1f} '
              f'total_steps {step_count} terminated_at {episode_step_count}')

    ## Save metrics for analysis
    if not load_checkpoint:
        
        np.save('tmp/Hopper-v5/rewards/step_rewards.npy', np.array([score_history, episode_steps]))
        np.save('tmp/Hopper-v5/analysis/temperature.npy', agent.temper_history)
        np.save('tmp/Hopper-v5/analysis/Q1_mean.npy', agent.q1_mean_history)
        np.save('tmp/Hopper-v5/analysis/Q2_mean.npy', agent.q2_mean_history)
        np.save('tmp/Hopper-v5/analysis/critic_loss.npy', agent.critic_loss_history)
        np.save('tmp/Hopper-v5/analysis/actor_loss.npy', agent.actor_loss_history)
        np.save('tmp/Hopper-v5/analysis/value_loss.npy', agent.value_loss_history)
    env.close()