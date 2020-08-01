import torchvision.transforms as T
import numpy as np
import torch
from StackedStates import StackedStates
from itertools import count
import random
import math

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
        _, reward, done, info = env.step(action)

        print('reward: {}, info: {}, done: {}'.format(reward, info, done))

        if done or t > 1000:
            env.close()
            break


if __name__ == '__main__':
    import gym
    import matplotlib.pyplot as plt
    # from torchvision.utils import make_grid

    # game = 'BreakoutNoFrameskip-v4'
    # game = 'SpaceInvadersNoFrameskip-v4'
    game = 'BreakoutDeterministic-v4'
    env = gym.make(game)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    print(f'env: {env}, action_space: {env.action_space}, observation_space: {env.observation_space}')

    n_actions = env.action_space.n
    print("Action Space Size: ", n_actions)

    stackedstates = StackedStates()
    env.reset()

    fig, axs = plt.subplots(5, 5)
    for i, ax in enumerate(axs.flat):
        state = get_state(env, stackedstates, device)
        _, _, _, _ = env.step(env.action_space.sample())

        flat_state = torch.cat([state[0, x, :, :] for x in range(4)], dim=1)
        ax.imshow(flat_state)
        ax.set_title(i)
    fig.suptitle(f'{state.shape}')
    plt.tight_layout()
    plt.show()

    interactive_play(env)
