import retro
import time
import numpy as np
from gym import Env
from gym.spaces import MultiBinary, Box, Discrete
import cv2
from matplotlib import pyplot as plt
import os
import optuna
from sb3_contrib import QRDQN
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

os.system("python -m retro.import ./custom_integrations")  

LOG_DIR = './logs/QRDQN'
OPT_DIR = './opt/QRDQN'
CHECKPOINT_DIR = './train/QRDQN/'

class StreetFighter(Env):
    def __init__(self,game_state='Champion.Level1.RyuVsGuile.state',record=False):
        super().__init__()
        self.observation_space = Box(low=0,high=255,shape=(84,84,1), dtype=np.uint8)
        self.action_space = Discrete(2**12)
        self.total_matches_won = 0
        self.total_enemy_matches_won = 0
        self.prev_matches_won = 0
        self.prev_enemy_matches_won = 0
        self.record = record
        # if record:
        #     self.game = retro.make(game='StreetFighterII-Champion', record='.',state=game_state,use_restricted_actions=retro.Actions.DISCRETE)
        # else:
        self.game = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', state=game_state,use_restricted_actions=retro.Actions.DISCRETE)


    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        obs_orig = obs
        obs = self.preprocess(obs)
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs
        if self.prev_matches_won > info['matches_won']:
            self.prev_matches_won = info['matches_won']
        if self.prev_enemy_matches_won > info['enemy_matches_won']:
            self.prev_enemy_matches_won = info['enemy_matches_won']
        self.total_matches_won = self.total_matches_won + info['matches_won'] - self.prev_matches_won
        self.total_enemy_matches_won = self.total_enemy_matches_won + info['enemy_matches_won'] - self.prev_enemy_matches_won
        self.prev_matches_won = info['matches_won']
        self.prev_enemy_matches_won = info['enemy_matches_won']
        reward = info['score'] - self.score
        self.score = info['score']
        self.obs_orig = obs_orig
        return frame_delta, reward, done, info
        
    def get_obs_orig(self):
        return self.obs_orig

    def render(self,*args,**kwargs):
        self.game.render()

    def reset(self):
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        self.score = 0
        self.prev_matches_won = 0
        self.prev_enemy_matches_won = 0
        return obs

    def preprocess(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84,84), cv2.INTER_CUBIC)
        channels = np.reshape(resize, (84,84,1))
        return channels

    def close(self):
        self.game.close()

def optimize(trial):
    return {
        'learning_rate':trial.suggest_loguniform('learning_rate',1e-5,1e-4),
        'gamma':trial.suggest_loguniform('gamma', 0.8,0.9999),
        'tau':trial.suggest_loguniform('tau', 0.001,0.01),
    }

def optimize_agent(trial):
        model_params = optimize(trial)
        env = StreetFighter()
        env = Monitor(env,LOG_DIR)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env,4,channels_order='last')

        model = QRDQN("CnnPolicy",env,tensorboard_log=LOG_DIR,verbose=2,batch_size=256,buffer_size=80000, **model_params) # cnn policy uses conv neural net for 

        # model = QRDQN("CnnPolicy",env,gradient_steps=-1,tensorboard_log=LOG_DIR,verbose=2,batch_size=256,buffer_size=80000, **model_params) # cnn policy uses conv neural net for 
        model.learn(total_timesteps=80000, progress_bar=True)

        mean_reward, _ = evaluate_policy(model,env,n_eval_episodes=5)
        env.close()
        
        SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(trial.number))
        model.save(SAVE_PATH)
        print(mean_reward)
        
        return mean_reward

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback,self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}_'.format(self.n_calls))
            self.model.save(model_path)
            self.logger.record('matches_won', self.training_env.get_attr('total_matches_won')[0])
            self.logger.record('enemy_matches_won', self.training_env.get_attr('total_enemy_matches_won')[0])

        return True


callback = TrainAndLoggingCallback(check_freq=10000,save_path=CHECKPOINT_DIR)
env = StreetFighter()
env = Monitor(env,LOG_DIR)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env,4,channels_order='last')
model_params = {}
model_params['learning_rate'] = 2.30575296217135e-05
model_params['gamma'] = 0.913579139196244
model_params['tau'] = 0.004267004883712244
model = QRDQN("CnnPolicy",env,tensorboard_log=LOG_DIR,verbose=2,batch_size=256,buffer_size=80000, **model_params) # cnn policy uses conv neural net for 
model.load(os.path.join(OPT_DIR, 'trial_1_best_model.zip'))
model.learn(total_timesteps=1000000,callback=callback, progress_bar=True)
# can increase training time and learning rate later