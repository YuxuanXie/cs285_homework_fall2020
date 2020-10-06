import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO_ return the action that maxinmizes the Q-value 
        # at the current observation as the output
        # import pdb; pdb.set_trace()

        actions = np.argmax(self.critic.qa_values(observation))

        return actions.squeeze()
