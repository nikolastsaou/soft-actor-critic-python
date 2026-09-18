# Soft Actor-Critic for MuJoCo Control Tasks

This project implements **Soft Actor-Critic (SAC)**, an off-policy deep
reinforcement-learning algorithm for continuous-action environments. The agent
learns a policy by interacting with a Gymnasium environment, storing
transitions in a replay buffer, and repeatedly updating neural networks from
sampled batches.

The implementation is tested with the following MuJoCo environments:

- `InvertedPendulum-v4`
- `HalfCheetah-v4`
- `Hopper-v5`

The goal is to train an agent that controls each simulated body and maximizes
the cumulative reward. During evaluation, the saved policy acts
deterministically and the environment can be rendered in a window.

## What is implemented?

The SAC agent in `Soft_Actor_Critic.py` contains:

- an actor network that samples bounded continuous actions;
- two critic networks that estimate Q-values;
- a value network and a slowly updated target-value network;
- a replay buffer for off-policy learning;
- reward scaling and soft target-network updates;
- optional automatic entropy-temperature tuning;
- model checkpoint saving/loading and training diagnostics.

`Networks.py` defines the actor, critic, and value networks. `ReplayBuffer.py`
stores state transitions and returns random training batches.

## Project layout

```text
Source_Code/
|-- main_sac.py                    Generic configurable training script
|-- main_sac_cheetah.py            HalfCheetah-v4 experiment
|-- main_sac_hopper.py             Hopper-v5 experiment
|-- main_sac_Inverted_Pendulum.py  InvertedPendulum-v4 experiment
|-- Soft_Actor_Critic.py           SAC agent and learning updates
|-- Networks.py                    Actor, critic, and value networks
|-- ReplayBuffer.py                Experience replay buffer
`-- tmp/<environment>/              Checkpoints, rewards, and analysis data
```

The `tmp` directory already contains some pretrained checkpoints and saved
training outputs from previous runs.

## Installation

Python **3.10** is recommended. From the project root, install the required
packages with:

```bash
pip install -r requirements.txt
```

The dependencies are PyTorch, Gymnasium, and the Gymnasium MuJoCo extras.
MuJoCo environments may also require the native MuJoCo dependencies supported
by the installed Gymnasium version.

## Training

Run commands from the `Source_Code` directory because the scripts use relative
paths such as `tmp/Hopper-v5/model`:

```bash
cd Source_Code
python main_sac_hopper.py
python main_sac_cheetah.py
python main_sac_Inverted_Pendulum.py
```

Alternatively, edit `env_name` and the hyperparameters in `main_sac.py`, then
run:

```bash
python main_sac.py
```

The environment-specific scripts use different training configurations. For
example, the Hopper experiment trains for up to one million environment steps,
while the other scripts use a fixed number of episodes. The scripts print each
episode's reward, a moving average, and selected network losses while training.

To adapt the project to another task, choose a Gymnasium environment with a
continuous action space and update the environment name, training duration,
hyperparameters, and checkpoint directory in the relevant entry point.

## Evaluation with a saved model

1. Open the entry-point script used for the desired environment.
2. Set `load_checkpoint = True`.
3. Make sure `chkpt_dir` points to the directory containing the checkpoint
	 files, then run the script from `Source_Code` again.

When checkpoint loading is enabled, the script loads the actor, both critics,
the value network, and the target-value network. It uses deterministic actions
and opens a Gymnasium render window where supported. Training updates and new
checkpoints are skipped in this mode.

The checkpoint files are named `actor_sac`, `critic_1_sac`, `critic_2_sac`,
`value_sac`, and `target_value_sac`.

## Saved results

After training, each experiment writes files below
`tmp/<environment>/`:

- `model/`: checkpoints created by the current run;
- `model_pretrained/`: included pretrained checkpoints, when available;
- `rewards/step_rewards.npy`: episode rewards (and episode step totals in the
	Hopper script);
- `analysis/temperature.npy`: entropy temperature history when automatic
	tuning is enabled;
- `analysis/Q1_mean.npy` and `analysis/Q2_mean.npy`: mean critic estimates;
- `analysis/actor_loss.npy`, `critic_loss.npy`, and `value_loss.npy`:
	optimization-loss histories.

The analysis and reward files are NumPy arrays and can be loaded with
`numpy.load` for plotting or comparison between environments.
