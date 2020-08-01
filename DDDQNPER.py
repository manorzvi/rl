from itertools import count
import gym

import torch
import torch.optim as optim
import torch.nn.functional as F

from StackedStates import StackedStates
from utils import get_state, get_config
from DDDQN import DDDQN
from PrioritizedExperienceReplay import PER


class DDDQNPER(DDDQN):

    def __init__(self, h, w, n_action, device, env_id, loss_func, optimizer_func,
                 exp_rep_capacity=100000, exp_rep_pretrain_size=100000,
                 batch_size=64, episodes=2000, target_update_interval=10, save_model_interval=100,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.00001, lr=0.00025, gamma=0.99, logs_dir='logs', ckpt_dir='models'):

        super(DDDQNPER, self).__init__(h, w, n_action, device, env_id, loss_func, optimizer_func, exp_rep_capacity,
                                       exp_rep_pretrain_size, batch_size, episodes, target_update_interval,
                                       save_model_interval, eps_start, eps_end, eps_decay, lr, gamma, logs_dir,
                                       ckpt_dir)
        print('[I] - Override Regular Experience Replay with Prioritized Experience Replay ... ', end='')
        self.memory = PER(capacity=exp_rep_capacity)
        print('Done.')

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
        tree_idx, batch, ISWeights = self.memory.sample(self.batch_size)

        states      = torch.cat([each.state         for each in batch], dim=0)
        next_states = torch.cat([each.next_state    for each in batch], dim=0)
        actions     = torch.cat([each.action        for each in batch], dim=0).unsqueeze(1)
        rewards     = torch.cat([each.reward        for each in batch], dim=0).unsqueeze(1)
        dones       = torch.cat([each.done          for each in batch], dim=0).float().unsqueeze(1)

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

        ISWeights = torch.from_numpy(ISWeights)
        loss = self.loss_func(Q_state_action, TD_target, reduction='none') * ISWeights
        loss = torch.mean(loss)

        abs_errors = torch.abs(TD_target - Q_state_action).detach().numpy()

        self.writer.add_scalar('Loss', loss.item(), global_step=self.steps_done)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.online_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.memory.batch_update(tree_idx, abs_errors)

        return loss.item()


if __name__ == '__main__':
    parser = get_config()
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

    model = DDDQNPER(h=screen_height, w=screen_width, n_action=n_action, device=device, env_id=args.env_id,
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

