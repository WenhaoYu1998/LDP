# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import utils
from sac_ae_ratio import  Actor, Critic, LOG_FREQ
from transition_model import make_transition_model


class AMBSRatioAgent(object):
    """Bisimulation metric algorithm."""
    def __init__(
        self,
        obs_shape,
        vector_state_shape,
        action_shape,
        device,
        transition_model_type,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        encoder_stride=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_type='pixel',
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_latent_lambda=0.0,
        decoder_weight_lambda=0.0,
        num_layers=4,
        num_filters=32,
        bisim_coef=0.5,
        sep_rew_dyn=False,
        sep_rew_ratio=0.5,
        adaptive_ratio=False,
        deep_metric=False,
    ):
        print(__file__)
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.encoder_feature_dim = encoder_feature_dim

        self.actor = Actor(
            obs_shape, vector_state_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic = Critic(
            obs_shape, vector_state_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic_target = Critic(
            obs_shape, vector_state_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())


        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, vector_state):
        with torch.no_grad():
            # obs = torch.FloatTensor(obs).to(self.device)
            obs = torch.tensor(obs, device=self.device, dtype=torch.float)
            obs = obs.unsqueeze(0)
            vector_state = torch.tensor(vector_state, device=self.device, dtype=torch.float)
            vector_state = vector_state.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, vector_state, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def select_action_batch(self, obs, vector_state):
        with torch.no_grad():
            # obs = torch.FloatTensor(obs).to(self.device)
            obs = torch.tensor(obs, device=self.device, dtype=torch.float)
            vector_state = torch.tensor(vector_state, device=self.device, dtype=torch.float)
            mu, _, _, _ = self.actor(
                obs, vector_state, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().flatten(start_dim=1).data.numpy()

    def sample_action(self, obs, vector_state):
        with torch.no_grad():
            # obs = torch.FloatTensor(obs).to(self.device)
            obs = torch.tensor(obs, device=self.device, dtype=torch.float)
            obs = obs.unsqueeze(0)
            vector_state = torch.tensor(vector_state, device=self.device, dtype=torch.float)
            vector_state = vector_state.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, vector_state, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def sample_action_batch(self, obs, vector_state):
        with torch.no_grad():
            # obs = torch.FloatTensor(obs).to(self.device)
            obs = torch.tensor(obs, device=self.device, dtype=torch.float)
            vector_state = torch.tensor(vector_state, device=self.device, dtype=torch.float)
            mu, pi, _, _ = self.actor(obs, vector_state, compute_log_pi=False)
            return pi.cpu().data.numpy()

    def update_critic(self, obs, vector_state, action, reward, next_obs, next_vector_state, not_done, L, step):
        with torch.no_grad():

            _, policy_action, log_pi, _ = self.actor(next_obs, next_vector_state)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_vector_state, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, vector_state, action, detach_encoder=False)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)

        L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 20.)
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, vector_state, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, vector_state, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, vector_state, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()
        L.log('train_actor/loss', actor_loss, step)
        L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        #nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 20.)
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        L.log('train_alpha/loss', alpha_loss, step)
        L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, L, step):
        obs, vector_state, action, _, reward, next_obs, next_vector_state, not_done = replay_buffer.sample()
        
        L.log('train/batch_reward', reward.mean(), step)
        
        #self.update_critic(obs, action, reward, next_obs, not_done, aug_obs, aug_next_obs, L, step)
        self.update_critic(obs, vector_state, action, reward, next_obs, next_vector_state, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, vector_state, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

            
    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic_target.state_dict(), '%s/critic_target_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.log_alpha,
            '%s/log_alpha_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )