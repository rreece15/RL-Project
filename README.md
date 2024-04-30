Steps to Run Code:

1. Run on python 3.7 using conda or a virtual env
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