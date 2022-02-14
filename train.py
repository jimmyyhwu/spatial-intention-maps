# Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import argparse
import random
import sys
from collections import namedtuple
from pathlib import Path

# Prevent numpy from using up all cpu
import os
os.environ['MKL_NUM_THREADS'] = '1'  # pylint: disable=wrong-import-position

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import smooth_l1_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils


torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.buffer)

class TransitionTracker:
    def __init__(self, initial_state):
        self.num_buffers = len(initial_state)
        self.prev_state = initial_state
        self.prev_action = [[None for _ in g] for g in self.prev_state]

    def update_action(self, action):
        for i, g in enumerate(action):
            for j, a in enumerate(g):
                if a is not None:
                    self.prev_action[i][j] = a

    def update_step_completed(self, reward, state, done):
        transitions_per_buffer = [[] for _ in range(self.num_buffers)]
        for i, g in enumerate(state):
            for j, s in enumerate(g):
                if s is not None or done:
                    if self.prev_state[i][j] is not None:
                        transition = (self.prev_state[i][j], self.prev_action[i][j], reward[i][j], s)
                        transitions_per_buffer[i].append(transition)
                    self.prev_state[i][j] = s
        return transitions_per_buffer

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Meters:
    def __init__(self):
        self.meters = {}

    def get_names(self):
        return self.meters.keys()

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def update(self, name, val):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(val)

    def avg(self, name):
        return self.meters[name].avg

def train(cfg, policy_net, target_net, optimizer, batch, transform_fn, discount_factor):
    state_batch = torch.cat([transform_fn(s) for s in batch.state]).to(device)  # (32, 4, 96, 96)
    action_batch = torch.tensor(batch.action, dtype=torch.long).to(device)  # (32,)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)  # (32,)
    non_final_next_states = torch.cat([transform_fn(s) for s in batch.next_state if s is not None]).to(device, non_blocking=True)  # (<=32, 4, 96, 96)

    output = policy_net(state_batch)  # (32, 2, 96, 96)
    state_action_values = output.view(cfg.batch_size, -1).gather(1, action_batch.unsqueeze(1)).squeeze(1)  # (32,)
    next_state_values = torch.zeros(cfg.batch_size, dtype=torch.float32, device=device)  # (32,)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool, device=device)  # (32,)

    if cfg.use_double_dqn:
        with torch.no_grad():
            best_action = policy_net(non_final_next_states).view(non_final_next_states.size(0), -1).max(1)[1].view(non_final_next_states.size(0), 1)  # (<=32, 1)
            next_state_values[non_final_mask] = target_net(non_final_next_states).view(non_final_next_states.size(0), -1).gather(1, best_action).view(-1)  # (<=32,)
    else:
        next_state_values[non_final_mask] = target_net(non_final_next_states).view(non_final_next_states.size(0), -1).max(1)[0].detach()  # (<=32,)

    expected_state_action_values = (reward_batch + discount_factor * next_state_values)  # (32,)
    td_error = torch.abs(state_action_values - expected_state_action_values).detach()  # (32,)

    loss = smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    if cfg.grad_norm_clipping is not None:
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), cfg.grad_norm_clipping)
    optimizer.step()

    train_info = {}
    train_info['td_error'] = td_error.mean().item()
    train_info['loss'] = loss.item()

    return train_info

def train_intention(intention_net, optimizer, batch, transform_fn):
    # Expects last channel of the state representation to be the ground truth intention map
    state_batch = torch.cat([transform_fn(s[:, :, :-1]) for s in batch.state]).to(device)  # (32, 4 or 5, 96, 96)
    target_batch = torch.cat([transform_fn(s[:, :, -1:]) for s in batch.state]).to(device)  # (32, 1, 96, 96)

    output = intention_net(state_batch)  # (32, 2, 96, 96)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(output, target_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_info = {}
    train_info['loss_intention'] = loss.item()

    return train_info

def main(cfg):
    # Set up logging and checkpointing
    log_dir = Path(cfg.log_dir)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    print('log_dir: {}'.format(log_dir))
    print('checkpoint_dir: {}'.format(checkpoint_dir))

    # Create environment
    kwargs = {}
    if cfg.show_gui:
        import matplotlib  # pylint: disable=import-outside-toplevel
        matplotlib.use('agg')
    if cfg.use_predicted_intention:  # Enable ground truth intention map during training only
        kwargs['use_intention_map'] = True
        kwargs['intention_map_encoding'] = 'ramp'
    env = utils.get_env_from_cfg(cfg, **kwargs)

    robot_group_types = env.get_robot_group_types()
    num_robot_groups = len(robot_group_types)

    # Policy
    policy = utils.get_policy_from_cfg(cfg, train=True)

    # Optimizers
    optimizers = []
    for i in range(num_robot_groups):
        optimizers.append(optim.SGD(policy.policy_nets[i].parameters(), lr=cfg.learning_rate, momentum=0.9, weight_decay=cfg.weight_decay))
    if cfg.use_predicted_intention:
        optimizers_intention = []
        for i in range(num_robot_groups):
            optimizers_intention.append(optim.SGD(policy.intention_nets[i].parameters(), lr=cfg.learning_rate, momentum=0.9, weight_decay=cfg.weight_decay))

    # Replay buffers
    replay_buffers = []
    for _ in range(num_robot_groups):
        replay_buffers.append(ReplayBuffer(cfg.replay_buffer_size))

    # Resume if applicable
    start_timestep = 0
    episode = 0
    if cfg.checkpoint_path is not None:
        checkpoint = torch.load(cfg.checkpoint_path)
        start_timestep = checkpoint['timestep']
        episode = checkpoint['episode']
        for i in range(num_robot_groups):
            optimizers[i].load_state_dict(checkpoint['optimizers'][i])
            replay_buffers[i] = checkpoint['replay_buffers'][i]
        if cfg.use_predicted_intention:
            for i in range(num_robot_groups):
                optimizers_intention[i].load_state_dict(checkpoint['optimizers_intention'][i])
        print("=> loaded checkpoint '{}' (timestep {})".format(cfg.checkpoint_path, start_timestep))

    # Target nets
    target_nets = policy.build_policy_nets()
    for i in range(num_robot_groups):
        target_nets[i].load_state_dict(policy.policy_nets[i].state_dict())
        target_nets[i].eval()

    # Logging
    train_summary_writer = SummaryWriter(log_dir=str(log_dir / 'train'))
    visualization_summary_writer = SummaryWriter(log_dir=str(log_dir / 'visualization'))
    meters = Meters()

    state = env.reset()
    transition_tracker = TransitionTracker(state)
    learning_starts = np.round(cfg.learning_starts_frac * cfg.total_timesteps).astype(np.uint32)
    total_timesteps_with_warm_up = learning_starts + cfg.total_timesteps
    for timestep in tqdm(range(start_timestep, total_timesteps_with_warm_up), initial=start_timestep, total=total_timesteps_with_warm_up, file=sys.stdout):
        # Select an action for each robot
        exploration_eps = 1 - (1 - cfg.final_exploration) * min(1, max(0, timestep - learning_starts) / (cfg.exploration_frac * cfg.total_timesteps))
        if cfg.use_predicted_intention:
            use_ground_truth_intention = max(0, timestep - learning_starts) / cfg.total_timesteps <= cfg.use_predicted_intention_frac
            action = policy.step(state, exploration_eps=exploration_eps, use_ground_truth_intention=use_ground_truth_intention)
        else:
            action = policy.step(state, exploration_eps=exploration_eps)
        transition_tracker.update_action(action)

        # Step the simulation
        state, reward, done, info = env.step(action)

        # Store in buffers
        transitions_per_buffer = transition_tracker.update_step_completed(reward, state, done)
        for i, transitions in enumerate(transitions_per_buffer):
            for transition in transitions:
                replay_buffers[i].push(*transition)

        # Reset if episode ended
        if done:
            state = env.reset()
            transition_tracker = TransitionTracker(state)
            episode += 1

        # Train networks
        if timestep >= learning_starts and (timestep + 1) % cfg.train_freq == 0:
            all_train_info = {}
            for i in range(num_robot_groups):
                batch = replay_buffers[i].sample(cfg.batch_size)
                train_info = train(cfg, policy.policy_nets[i], target_nets[i], optimizers[i], batch, policy.apply_transform, cfg.discount_factors[i])

                if cfg.use_predicted_intention:
                    train_info_intention = train_intention(policy.intention_nets[i], optimizers_intention[i], batch, policy.apply_transform)
                    train_info.update(train_info_intention)

                for name, val in train_info.items():
                    all_train_info['{}/robot_group_{:02}'.format(name, i + 1)] = val

        # Update target networks
        if (timestep + 1) % cfg.target_update_freq == 0:
            for i in range(num_robot_groups):
                target_nets[i].load_state_dict(policy.policy_nets[i].state_dict())

        ################################################################################
        # Logging

        # Meters
        if timestep >= learning_starts and (timestep + 1) % cfg.train_freq == 0:
            for name, val in all_train_info.items():
                meters.update(name, val)

        if done:
            for name in meters.get_names():
                train_summary_writer.add_scalar(name, meters.avg(name), timestep + 1)
            meters.reset()

            train_summary_writer.add_scalar('steps', info['steps'], timestep + 1)
            train_summary_writer.add_scalar('total_cubes', info['total_cubes'], timestep + 1)
            train_summary_writer.add_scalar('episodes', episode, timestep + 1)

            for i in range(num_robot_groups):
                for name in ['cumulative_cubes', 'cumulative_distance', 'cumulative_reward', 'cumulative_robot_collisions']:
                    train_summary_writer.add_scalar('{}/robot_group_{:02}'.format(name, i + 1), np.mean(info[name][i]), timestep + 1)

            # Visualize Q-network outputs
            if timestep >= learning_starts:
                random_state = [[random.choice(replay_buffers[i].buffer).state] for _ in range(num_robot_groups)]
                _, info = policy.step(random_state, debug=True)
                for i in range(num_robot_groups):
                    visualization = utils.get_state_output_visualization(random_state[i][0], info['output'][i][0]).transpose((2, 0, 1))
                    visualization_summary_writer.add_image('output/robot_group_{:02}'.format(i + 1), visualization, timestep + 1)
                    if cfg.use_predicted_intention:
                        visualization_intention = utils.get_state_output_visualization(
                            random_state[i][0],
                            np.stack((random_state[i][0][:, :, -1], info['output_intention'][i][0]), axis=0)  # Ground truth and output
                        ).transpose((2, 0, 1))
                        visualization_summary_writer.add_image('output_intention/robot_group_{:02}'.format(i + 1), visualization_intention, timestep + 1)

        ################################################################################
        # Checkpointing

        if (timestep + 1) % cfg.checkpoint_freq == 0 or timestep + 1 == total_timesteps_with_warm_up:
            if not checkpoint_dir.exists():
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save policy
            policy_filename = 'policy_{:08d}.pth.tar'.format(timestep + 1)
            policy_path = checkpoint_dir / policy_filename
            policy_checkpoint = {
                'timestep': timestep + 1,
                'state_dicts': [policy.policy_nets[i].state_dict() for i in range(num_robot_groups)],
            }
            if cfg.use_predicted_intention:
                policy_checkpoint['state_dicts_intention'] = [policy.intention_nets[i].state_dict() for i in range(num_robot_groups)]
            torch.save(policy_checkpoint, str(policy_path))

            # Save checkpoint
            checkpoint_filename = 'checkpoint_{:08d}.pth.tar'.format(timestep + 1)
            checkpoint_path = checkpoint_dir / checkpoint_filename
            checkpoint = {
                'timestep': timestep + 1,
                'episode': episode,
                'optimizers': [optimizers[i].state_dict() for i in range(num_robot_groups)],
                'replay_buffers': [replay_buffers[i] for i in range(num_robot_groups)],
            }
            if cfg.use_predicted_intention:
                checkpoint['optimizers_intention'] = [optimizers_intention[i].state_dict() for i in range(num_robot_groups)]
            torch.save(checkpoint, str(checkpoint_path))

            # Save updated config file
            cfg.policy_path = str(policy_path)
            cfg.checkpoint_path = str(checkpoint_path)
            utils.save_config(log_dir / 'config.yml', cfg)

            # Remove old checkpoint
            checkpoint_paths = list(checkpoint_dir.glob('checkpoint_*.pth.tar'))
            checkpoint_paths.remove(checkpoint_path)
            for old_checkpoint_path in checkpoint_paths:
                old_checkpoint_path.unlink()

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path')
    config_path = parser.parse_args().config_path
    if config_path is None:
        if sys.platform == 'darwin':
            config_path = 'config/local/lifting_4-small_empty-local.yml'
        else:
            config_path = utils.select_run()
    if config_path is not None:
        config_path = utils.setup_run(config_path)
        main(utils.load_config(config_path))
