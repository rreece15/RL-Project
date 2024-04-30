# Deep Reinforcement Learning for Street Fighter II Champion Edition

Corresponding to the growth of reinforcement learning, the field has blossomed in theoretical and practical applications. Notably, reinforcement learning has seen profound success in the gaming industry. Building upon this trend, our study investigates the applicability of reinforcement learning within fighting games, capitalizing on its proficiency in games with lower state spaces (among other features) like Atari games. Thus, we choose a fitting, though more complex, fighting game, Street Fighter II Champions Edition, to explore various reinforcement learning models, iterating on existing studies with a more thorough analysis of applied algorithms and variation in-game settings. By applying Deep Q-Network (DQN) and Quantile Regression DQN (QRDQN) to a baseline of Proximal Policy Optimization (PPO), we find the relative performance of approaches. Furthermore, we delve into character-specific analyses to offer insights into optimal character choices for competitive players. Ultimately, this paper investigates applying deep reinforcement learning models to the iconic fighting game, Street Fighter II Champion Edition, aiming to understand the efficacy of different RL algorithms in this challenging domain and advancing the understanding of their practical utility for future research endeavors.

## Links

Presentation Link: https://youtu.be/7D2RxDPWJ_8

Other Model Results: https://youtu.be/uqJQnNCKViQ

## Instructions

State Creation:

1. To create additional states, the ROM used for the Gym-Retro environment was imported to the Gym-Retro site-package within the Anaconda environment.
2. The Gym-Retro Integration UI was used to save states by playing until the desired step within the game, then saving the state.
3. For additional assistance, see the following discussion on GitHub: https://github.com/openai/retro/issues/33

Steps to Run Code:

1. Run on python 3.8 using conda or a virtual env; if using Jupyter Notebooks in VsCode, you may first have to create an environment with Python 3.8 and downgrade
2. Retrieve the necessary packages from the requirements file with the following command while in the python 3.7 environment

pip install -r requirements.txt

For Training:

1. Import packages and game
2. Choose state for starting character from available states in /custom_integrations
3. Compile the SF2 environment class
4. Set the optimization and log paths
5. Change total timesteps in optimize_agent() function
6. Compile optimization functions
7. Run Optimization (We ran for 10 trials at 80,000 steps each)
8. Set training path
9. Load the model with the highest value
10. Compile callback function
11. Set parameters equal to the best trial
12. Change number of training steps
13. Run training (We ran for 1,000,000 steps)

For Evaluation:

1. Import packages and game
2. Choose state for starting character from available states in /custom_integrations
3. Compile the SF2 environment class
4. Set the optimization, training, and log paths
5. Load model from the train folder at the bottom of its corresponding notebook
6. Run run loop at the bottom of the notebook

Results:
Folders in the train directory contain the highest reward model, the model at 1 million step, and the final trained model
The data directory contains individual win rates, rewards, and length values for different runs
