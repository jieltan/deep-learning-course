import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        ##### TODO ######
        ### Complete definition
        # actor
        self.fc1 = nn.Linear(4,128)
        self.fc_a = nn.Linear(128,2)
        self.sm = nn.SoftMax()
        self.fc_c = nn.Linear(128,1)
        self.relu_a = nn.ReLU()
        self.relu_c = nn.ReLU()

    def forward(self, x):
        ##### TODO ######
        ### Complete definition
        h1 = self.fc1(x)
        a1 = self.fc_a(self.relu_a(h1))
        c1 = self.fc_c(self.relu_c(h1))
        a2 = self.sm(a1)
        return a1, c1

model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    return action

def sample_episode():

    state, ep_reward = env.reset(), 0
    episode = []

    for t in range(1, 10000):  # Run for a max of 10k steps

        action = select_action(state)

        # Perform action
        next_state, reward, done, _ = env.step(action.item())

        episode.append((state, action, reward))
        state = next_state

        ep_reward += reward

        if args.render:
            env.render()

        if done:
            break

    return episode, ep_reward

def compute_losses(episode):

    ####### TODO #######
    #### Compute the actor and critic losses
    dis_reward = []
    running_add = 0
    for i reversed(range(len(episode))):
        running_add = arg.gamma * running_add + episode[2][i]
        dis_reward.insert(0, running_add)
    dis_reward = torch.tensor(dis_reward)
    dis_reward = (dis_reward - dis_reward.mean())/(dis_reward.std())
    logprob = []
    c_hist = []
    actor_loss, critic_loss = torch.tensor(0.), torch.tensor(0.)
    for i in range(len(episode)):
        a, c = model(episode[0][i])
        logprob.append(torch.log(a[episode[1][i]]))
        c_hist.append(c)
    for i range(len(episode)):
        adv = dis_reward[i] - c_hist[i].item()
        actor_loss = -1. * logprob * adv + actor_loss
        critic_loss += adv.pow(2)


    return actor_loss, critic_loss


def main():
    running_reward = 10
    for i_episode in count(1):

        episode, episode_reward = sample_episode()

        optimizer.zero_grad()

        actor_loss, critic_loss = compute_losses(episode)

        loss = actor_loss + critic_loss

        loss.backward()

        optimizer.step()

        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, episode_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, len(episode)))
            break


if __name__ == '__main__':
    main()
