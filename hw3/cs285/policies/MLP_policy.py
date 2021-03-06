import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions
from cs285.infrastructure import utils
from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers, size=self.size)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO_: get this from hw1 or hw2
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation)
        
        if self.discrete:
            return distributions.Categorical(self.logits_na(observation)).sample().detach().numpy()
        probs_out = self.mean_net(observation)
        action = probs_out + torch.exp(self.logstd) * torch.randn(probs_out.size())
        return action.detach().numpy()

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO_: get this from hw1 or hw2
        if self.discrete:
            return distributions.Categorical(self.logits_na(observation))
        return self.mean_net(observation)


#####################################################
#####################################################


class MLPPolicyAC(MLPPolicy):
    def update(self, observations, actions, adv_n=None):
        # TODO_: update the policy and return the loss
        # loss = TODO_
        # return loss.item()

        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(adv_n)

        # TODO_: compute the loss that should be optimized when training with policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
            # is the expectation over collected trajectories of:
            # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: you will want to use the `log_prob` method on the distribution returned
            # by the `forward` method
        # HINT3: don't forget that `optimizer.step()` MINIMIZES a loss

        if self.discrete :
            log_prob = self.forward(observations).log_prob(actions)
        else:
            log_prob = utils.multivariate_normal_diag(loc = self.forward(observations), scale_diag=torch.exp(self.logstd)).log_prob(actions)

        if self.nn_baseline:
            loss = torch.mean(log_prob * (torch.squeeze(self.baseline(observations)) - ptu.from_numpy(q_values)))
        else:
            loss = - 1.0 * torch.mean(log_prob * advantages)

        # import pdb; pdb.set_trace()


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # TODO_: optimize `loss` using `self.optimizer`
        # HINT: remember to `zero_grad` first
        # TODO_

        if self.nn_baseline:
            ## TODO_: normalize the q_values to have a mean of zero and a standard deviation of one
            ## HINT: there is a `normalize` function in `infrastructure.utils`
            targets = utils.normalize(q_values, 0, 1)
            targets = ptu.from_numpy(targets)

            ## TODO_: use the `forward` method of `self.baseline` to get baseline predictions
            baseline_predictions = torch.squeeze(self.baseline(observations))
            
            ## avoid any subtle broadcasting bugs that can arise when dealing with arrays of shape
            ## [ N ] versus shape [ N x 1 ]
            ## HINT: you can use `squeeze` on torch tensors to remove dimensions of size 1
            assert baseline_predictions.shape == targets.shape
            
            # TODO_: compute the loss that should be optimized for training the baseline MLP (`self.baseline`)
            # HINT: use `F.mse_loss`
            baseline_loss = self.baseline_loss(targets, baseline_predictions)

            # TODO_: optimize `baseline_loss` using `self.baseline_optimizer`
            # HINT: remember to `zero_grad` first
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()

        return loss.item()
