import torchvision.transforms as T
import numpy as np
import torch
from StackedStates import StackedStates
from itertools import count
import argparse

resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(),
                    T.Resize(64),
                    T.ToTensor()])


def get_state(env, stackedstates: StackedStates, device: torch.device) -> torch.Tensor:

    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    screen = resize(screen).squeeze()
    if not hasattr(stackedstates, 'stack'):
        stackedstates.reset(screen)
    stackedstates.push(screen)

    return stackedstates().unsqueeze(0).to(device)


def interactive_play(env):

    print(f'[I] - Interactive play mode.')

    env.reset()

    for t in count():
        env.render()
        action = int(input('Choose action: '))
        state, reward, done, info = env.step(action)

        print('reward: {}, info: {}, done: {}'.format(reward, info, done))

        if done or t > 1000:
            env.close()
            break


def get_config():

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

    return parser


if __name__ == '__main__':
    import gym
    import matplotlib.pyplot as plt
    # from torchvision.utils import make_grid

    # game = 'BreakoutNoFrameskip-v4'
    # game = 'SpaceInvadersNoFrameskip-v4'
    # game = 'BreakoutDeterministic-v4'
    game = 'CartPole-v0'
    env = gym.make(game)
    env = env.unwrapped

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    print(f'env: {env}, action_space: {env.action_space}, observation_space: {env.observation_space}')

    n_actions = env.action_space.n
    print("Action Space Size: ", n_actions)

    stackedstates = StackedStates()
    env.reset()

    # fig, axs = plt.subplots(5, 5)
    # for i, ax in enumerate(axs.flat):
    #     state = get_state(env, stackedstates, device)
    #     _, _, _, _ = env.step(env.action_space.sample())
    #
    #     flat_state = torch.cat([state[0, x, :, :] for x in range(4)], dim=1)
    #     ax.imshow(flat_state)
    #     ax.set_title(i)
    # fig.suptitle(f'{state.shape}')
    # plt.tight_layout()
    # plt.show()

    interactive_play(env)
