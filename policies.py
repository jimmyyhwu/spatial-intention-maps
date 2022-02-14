import random

import numpy as np
import torch
from torchvision import transforms

import networks
from envs import VectorEnv


class DQNPolicy:
    def __init__(self, cfg, train=False, random_seed=None):
        self.cfg = cfg
        self.robot_group_types = [next(iter(g.keys())) for g in self.cfg.robot_config]
        self.train = train
        if random_seed is not None:
            random.seed(random_seed)

        self.num_robot_groups = len(self.robot_group_types)
        self.transform = transforms.ToTensor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_nets = self.build_policy_nets()

        # Resume if applicable
        if self.cfg.checkpoint_path is not None:
            self.policy_checkpoint = torch.load(self.cfg.policy_path, map_location=self.device)
            for i in range(self.num_robot_groups):
                self.policy_nets[i].load_state_dict(self.policy_checkpoint['state_dicts'][i])
                if self.train:
                    self.policy_nets[i].train()
                else:
                    self.policy_nets[i].eval()
            print("=> loaded policy '{}'".format(self.cfg.policy_path))

    def build_policy_nets(self):
        policy_nets = []
        for robot_type in self.robot_group_types:
            num_output_channels = VectorEnv.get_num_output_channels(robot_type)
            policy_nets.append(torch.nn.DataParallel(
                networks.FCN(num_input_channels=self.cfg.num_input_channels, num_output_channels=num_output_channels)
            ).to(self.device))
        return policy_nets

    def apply_transform(self, s):
        return self.transform(s).unsqueeze(0)

    def step(self, state, exploration_eps=None, debug=False):
        if exploration_eps is None:
            exploration_eps = self.cfg.final_exploration

        action = [[None for _ in g] for g in state]
        output = [[None for _ in g] for g in state]
        with torch.no_grad():
            for i, g in enumerate(state):
                robot_type = self.robot_group_types[i]
                self.policy_nets[i].eval()
                for j, s in enumerate(g):
                    if s is not None:
                        s = self.apply_transform(s).to(self.device)
                        o = self.policy_nets[i](s).squeeze(0)
                        if random.random() < exploration_eps:
                            a = random.randrange(VectorEnv.get_action_space(robot_type))
                        else:
                            a = o.view(1, -1).max(1)[1].item()
                        action[i][j] = a
                        output[i][j] = o.cpu().numpy()
                if self.train:
                    self.policy_nets[i].train()

        if debug:
            info = {'output': output}
            return action, info

        return action

class DQNIntentionPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intention_nets = self.build_intention_nets()
        if self.cfg.checkpoint_path is not None:
            for i in range(self.num_robot_groups):
                self.intention_nets[i].load_state_dict(self.policy_checkpoint['state_dicts_intention'][i])
                if self.train:
                    self.intention_nets[i].train()
                else:
                    self.intention_nets[i].eval()
            print("=> loaded intention network '{}'".format(self.cfg.policy_path))

    def build_intention_nets(self):
        intention_nets = []
        for _ in range(self.num_robot_groups):
            intention_nets.append(torch.nn.DataParallel(
                networks.FCN(num_input_channels=(self.cfg.num_input_channels - 1), num_output_channels=1)
            ).to(self.device))
        return intention_nets

    def step_intention(self, state, debug=False):
        state_intention = [[None for _ in g] for g in state]
        output_intention = [[None for _ in g] for g in state]
        with torch.no_grad():
            for i, g in enumerate(state):
                self.intention_nets[i].eval()
                for j, s in enumerate(g):
                    if s is not None:
                        s_copy = s.copy()
                        s = self.apply_transform(s).to(self.device)
                        o = torch.sigmoid(self.intention_nets[i](s)).squeeze(0).squeeze(0).cpu().numpy()
                        state_intention[i][j] = np.concatenate((s_copy, np.expand_dims(o, 2)), axis=2)
                        output_intention[i][j] = o
                if self.train:
                    self.intention_nets[i].train()

        if debug:
            info = {'output_intention': output_intention}
            return state_intention, info

        return state_intention

    def step(self, state, exploration_eps=None, debug=False, use_ground_truth_intention=False):
        if self.train and use_ground_truth_intention:
            # Use the ground truth intention map
            return super().step(state, exploration_eps=exploration_eps, debug=debug)

        if self.train:
            # Remove ground truth intention map
            state_copy = [[None for _ in g] for g in state]
            for i, g in enumerate(state):
                for j, s in enumerate(g):
                    if s is not None:
                        state_copy[i][j] = s[:, :, :-1]
            state = state_copy

        # Add predicted intention map to state
        state = self.step_intention(state, debug=debug)
        if debug:
            state, info_intention = state

        action = super().step(state, exploration_eps=exploration_eps, debug=debug)

        if debug:
            action, info = action
            info['state_intention'] = state
            info['output_intention'] = info_intention['output_intention']
            return action, info

        return action
