import torch
import numpy as np
import torch.nn.functional as F

from pong_env import PongEnv

# TODO replace this class with your model
class MyModelClass(torch.nn.Module):
    
    def __init__(self):
        super(MyModelClass, self).__init__()

        self.number_of_actions = 3
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.relu1 = torch.nn.ReLU(inplace=True)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.fc1 = torch.nn.Linear(7, 50).float()
        self.fc2 = torch.nn.Linear(50, 50).float()
        self.fc3 = torch.nn.Linear(50, self.number_of_actions).float()
    
    def forward(self, x):
        # x = np.array(x)
        # x = Variable(torch.from_numpy(x))
        x = torch.tensor(x).type('torch.FloatTensor')
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)

        return out


# TODO fill out the methods of this class
class PongPlayer(object):

    def __init__(self, save_path, load=False):
        self.build_model()
        self.build_optimizer()
        self.save_path = save_path
        if load:
            self.load()

    def build_model(self):
        # TODO: define your model here
        # I would suggest creating another class that subclasses
        # torch.nn.Module. Then you can just instantiate it here.
        # your not required to do this but if you don't you should probably
        # adjust the load and save functions to work with the way you did it.
        self.model = MyModelClass()

    def build_optimizer(self):
        # TODO: define your optimizer here
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def get_action(self, state):
        # TODO: this method should return the output of your model
        indices = torch.argmax(self.model.forward(state), 0)
        return indices.item()

    def reset(self):
        # TODO: this method will be called whenever a game finishes
        # so if you create a model that has state you should reset it here
        # NOTE: this is optional and only if you need it for your model
        pass

    def load(self):
        state = torch.load(self.save_path)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])

    def save(self):
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(state, self.save_path)

    
def play_game(player, render=True):
    # call this function to run your model on the environment
    # and see how it does
    env = PongEnv()
    state = env.reset()
    action = player.get_action(state)
    done = False
    total_reward = 0
    while not done:
        next_state, reward, done, _ = env.step(action)
        if render:
            env.render()
        action = player.get_action(next_state)
        total_reward += reward
    
    env.close()

if __name__ == '__main__':
    player = PongPlayer('./')
    play_game(player)
    