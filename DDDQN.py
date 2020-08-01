import random
import math
import os
import numpy as np
from scipy.stats import entropy
from datetime import datetime
from itertools import count
import gym
import argparse
import time
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ExperienceReplay import ExperienceReplay, Transition
from StackedStates import StackedStates
from utils import get_state


class DDDQN:

    def __init__(self, h, w, n_action, device, env_id, loss_func, optimizer_func,
                 exp_rep_capacity=100000, exp_rep_pretrain_size=100000,
                 batch_size=64, episodes=2000, target_update_interval=10, save_model_interval=100,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.00001, lr=0.00025, gamma=0.99, logs_dir='logs', ckpt_dir='models'):

        self.online_net         = DDDQNet(h, w, n_action).to(device)
        self.target_net         = DDDQNet(h, w, n_action).to(device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.memory = ExperienceReplay(capacity=exp_rep_capacity)
        self.exp_rep_pretrain_size = exp_rep_pretrain_size

        self.eps_start          = eps_start
        self.eps_end            = eps_end
        self.eps_decay          = eps_decay
        self.steps_done         = 0

        self.device             = device
        self.n_action           = n_action

        self.ckpt_dir           = os.path.abspath(os.path.join(ckpt_dir, env_id))
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.logs_dir           = os.path.abspath(os.path.join(logs_dir, env_id))
        print(f'[I] - Models Directory: {self.ckpt_dir}\n'
              f'[I] - Logs Directory: {self.logs_dir}')
        self.writer = SummaryWriter(self.logs_dir)

        self.batch_size = batch_size
        self.episodes = episodes
        self.target_update_interval = target_update_interval
        self.save_model_interval = save_model_interval
        self.gamma = gamma

        self.loss_func          = loss_func
        self.optimizer          = optimizer_func(self.online_net.parameters(), lr=lr)

    def __str__(self):
        return '|' + '----------' + '|\n' + \
               '|' + 'Online Net' + '|' + '\n' + \
               '|' + '----------' + '|\n' + \
               str(self.online_net) + '\n' + \
               '|' + '----------' + '|\n' + \
               '|' + 'Target Net' + '|' + '\n' + \
               '|' + '----------' + '|\n' + \
               str(self.online_net) + '\n' + \
               '|' + '----------' + '|\n' + \
               '|' + '   Loss   ' + '|' + '\n' + \
               '|' + '----------' + '|\n' + \
               str(self.loss_func) + '\n' + \
               '|' + '---------' + '|\n' + \
               '|' + 'Optimizer' + '|' + '\n' + \
               '|' + '---------' + '|\n' + \
               str(self.optimizer)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end+(self.eps_start-self.eps_end)*math.exp(-self.steps_done*self.eps_decay)
        self.writer.add_scalar('Epsilon_Greedy_Threshold', eps_threshold, global_step=self.steps_done)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                action_index = torch.argmax(self.online_net(state), dim=1)
            return action_index.item()
        else:
            return torch.tensor([random.randrange(self.n_action)], device=self.device, dtype=torch.long).item()

    def load(self, path):
        print('[I] Load Model ... ', end='')
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        print('Done.')

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, episode):
        now = datetime.now()
        torch.save({
            'episode': episode,
            'model_state_dict': self.online_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.ckpt_dir, f'episode-{episode}__{now.strftime("%m-%d-%y_%H:%M:%S")}.pt'))

    def exp_rep_pretrain(self, env):
        i = 0

        print('Pretrain Filling Experience Replay Memory ... ', end='')
        while i < self.exp_rep_pretrain_size:

            # Initialize the environment and state
            stackedstates = StackedStates()
            env.reset()
            state = get_state(env, stackedstates, self.device)

            for t in count():

                i += 1
                action = env.action_space.sample()
                _, reward, done, _ = env.step(action)

                reward = torch.tensor([reward], device=self.device)
                done   = torch.tensor([done],   device=self.device)
                action = torch.tensor([action], device=self.device)

                # Observe new state
                next_state = get_state(env, stackedstates, self.device)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward, done)

                if done:
                    print("{} ".format(t + 1), end='')
                    break
                else:
                    # Move to the next state
                    state = next_state
        print('Done.')

    def optimize(self):
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        states      = torch.cat(batch.state)
        next_states = torch.cat(batch.next_state)
        actions     = torch.cat(batch.action).unsqueeze(1)
        rewards     = torch.cat(batch.reward).unsqueeze(1)
        dones       = torch.cat(batch.done).float().unsqueeze(1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        Q_state_action = self.online_net(states).gather(1, actions)

        self.writer.add_scalar('Mean Q', torch.mean(Q_state_action).detach().item(), global_step=self.steps_done)

        with torch.no_grad():
            # Use Policy Network to select the action to take at next_state (a') (action with the highest Q-value)
            Q_next_state_action = self.online_net(next_states)
            Q_next_state_action_argmax = torch.argmax(Q_next_state_action, dim=1, keepdim=True)
            # Use Target Network to calculate the Q_val of Q(s',a')
            Q_next_state_action_target = self.target_net(next_states)
            # Compute the expected TD Target
            TD_target = rewards + (1 - dones) * self.gamma * Q_next_state_action_target.gather(1, Q_next_state_action_argmax)

        loss = self.loss_func(Q_state_action, TD_target)

        self.writer.add_scalar('Loss', loss.item(), global_step=self.steps_done)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.online_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def train(self, env):

        for i_episode in range(self.episodes):

            # Initialize the environment and state
            stackedstates = StackedStates()
            env.reset()
            state = get_state(env, stackedstates, self.device)

            rewards = []
            losses = []
            actions = []

            for t in count():

                # Select and perform an action
                action = self.select_action(state)
                _, reward, done, _ = env.step(action)

                actions.append(action)
                rewards.append(reward)

                reward = torch.tensor([reward], device=self.device)
                done = torch.tensor([done], device=self.device)
                action = torch.tensor([action], device=self.device)

                # Observe new state
                next_state = get_state(env, stackedstates, self.device)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward, done)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization
                loss = self.optimize()
                losses.append(loss)

                if done:
                    action_entropy = entropy(np.histogram(actions, bins=self.n_action, density=True)[0])
                    self.writer.add_scalar('Episode Reward', sum(rewards), global_step=i_episode)
                    self.writer.add_scalar('Action Entropy', action_entropy, global_step=i_episode)
                    self.writer.add_scalar('Episode Length', t, global_step=i_episode)
                    print(f'Episode {i_episode} Done.'
                          f'Length: {t}.'
                          f'Total Reward: {sum(rewards)}.'
                          f'Avg Loss: {sum(losses) / len(losses)}.'
                          f'Action Entropy: {action_entropy}.')
                    break

            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.target_update_interval == 0:
                print('[I] - Update Traget Net ... ', end='')
                self.update_target()
                print('Done.')

            if i_episode % self.save_model_interval == 0:
                print('[I] - Save Model ... ', end='')
                self.save(i_episode)
                print('Done.')

        print('Complete')

        print('[I] - Save Model ... ', end='')
        self.save(i_episode)
        print('Done.')

    def play(self, env):

        print(f'[I] - Set Online Net to evaluation mode ... ', end='')
        self.online_net.eval()
        print('Done.')

        for i in range(5):

            stackedstates = StackedStates()
            env.reset()
            state = get_state(env, stackedstates, self.device)

            for t in count():
                env.render()
                # time.sleep(0.04)

                with torch.no_grad():
                    action = torch.argmax(self.online_net(state), dim=1).item()

                _, reward, done, _ = env.step(action)
                state = get_state(env, stackedstates, self.device)

                if done or t > 1000:
                    env.close()
                    break


class DDDQNet(nn.Module):

    def __init__(self, h, w, n_action):
        super(DDDQNet, self).__init__()

        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2, padding=0, dilation=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0, dilation=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=0, dilation=1)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_out(size, kernel_size=5, stride=2, padding=0, dilation=1):
            return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

        convw = conv2d_out(conv2d_out(conv2d_out(w)))
        convh = conv2d_out(conv2d_out(conv2d_out(h)))

        linear_input_size = convw * convh * 32

        # Value Function Head:
        self.value_fc = nn.Linear(linear_input_size, 512)
        self.value  = nn.Linear(512, 1)

        # Advantage Function
        self.advantage_fc = nn.Linear(linear_input_size, 512)
        self.advantage  = nn.Linear(512, n_action)

    def forward(self, x):
        x         = nn.ELU()(self.bn1(self.conv1(x)))
        x         = nn.ELU()(self.bn2(self.conv2(x)))
        x         = nn.ELU()(self.bn3(self.conv3(x)))
        value     = self.value(nn.ELU()(self.value_fc(    x.view(x.size(0), -1))))
        advantage = self.advantage(nn.ELU()(self.advantage_fc(x.view(x.size(0), -1))))
        avg_advantage = torch.mean(advantage, dim=1, keepdim=True)
        q = value + advantage - avg_advantage
        return q


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--experience_replay_capacity', '-exp_rep_cap',         type=int,   default=100000,
                        help="Size of the Experience Replay Memory")
    parser.add_argument('--experience_replay_pretrain_size', '-exp_rep_pre',    type=int,   default=100000,
                        help="Size of experiences to store before the training begins")
    parser.add_argument('--batch_size', '-bs',                                  type=int,   default=32)
    parser.add_argument('--episodes_number', '-epi_num',                        type=int,   default=1000,
                        help='Number of episodes to play')
    parser.add_argument('--target_update_interval', '-tar_updt_int',            type=int,   default=10,
                        help='Target Network update interval')
    parser.add_argument('--save_model_interval', '-save_mdl_int',                type=int,   default=10,
                        help='Online Network saving interval')
    parser.add_argument('--epsilon_start', '-eps_start',                        type=float, default=1.0,
                        help='Start value for Epsilon Greedy strategy')
    parser.add_argument('--epsilon_end', '-eps_end',                            type=float, default=0.01,
                        help='End value for Epsilon Greedy strategy')
    parser.add_argument('--epsilon_decay', '-eps_decay',                        type=float, default=0.00001,
                        help='Decay Rate for Epsilon Greedy strategy')
    parser.add_argument('--learning_rate', '-lr',                               type=float, default=0.00025,
                        help="Optimizer's Learning Rate")
    parser.add_argument('--gamma', '-gamma',                                    type=float, default=0.99,
                        help="Q Learning Discount Factor")
    parser.add_argument('--logs', '-logs',                                      type=str,   default='logs',
                        help="path to logs directory")
    parser.add_argument('--models', '-models',                                  type=str,   default='models',
                        help="path to models directory")
    parser.add_argument('--env_id', '-env_id',                                  type=str,   default='BreakoutNoFrameskip-v4',
                        help="OpenAI Gym Env ID")
    parser.add_argument('--path', '-path',                                      type=str,
                        help="Relative path to existing model")
    parser.add_argument('--load', '-load',                                      action='store_true', default=False,
                        help='Load existing model')
    parser.add_argument('--play', '-play',                                      action='store_true', default=False,
                        help='Play')
    parser.add_argument('--train', '-train',                                    action='store_true', default=False,
                        help='Train Model')

    args = parser.parse_args()

    print(args)

    env = gym.make(args.env_id)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    print(f'env: {env}, action_space: {env.action_space}, observation_space: {env.observation_space}')

    n_action = env.action_space.n
    print("Action Space Size: ", n_action)

    stackedstates = StackedStates()
    env.reset()
    init_state = get_state(env, stackedstates, device)
    _, _, screen_height, screen_width = init_state.shape

    loss_func = F.smooth_l1_loss
    optimizer_func = optim.Adam

    model = DDDQN(h=screen_height, w=screen_width, n_action=n_action, device=device, env_id=args.env_id,
                  loss_func=loss_func, optimizer_func=optimizer_func, exp_rep_capacity=args.experience_replay_capacity,
                  exp_rep_pretrain_size=args.experience_replay_pretrain_size,
                  batch_size=args.batch_size, episodes=args.episodes_number,
                  target_update_interval=args.target_update_interval, save_model_interval=args.save_model_interval,
                  eps_start=args.epsilon_start, eps_end=args.epsilon_end, eps_decay=args.epsilon_decay,
                  lr=args.learning_rate, gamma=args.gamma, logs_dir=args.logs, ckpt_dir=args.models)

    if args.load:
        model.load(args.path)

    print(model)

    if args.train:
        model.exp_rep_pretrain(env)

        model.train(env)

    if args.play:
        model.play(env)


