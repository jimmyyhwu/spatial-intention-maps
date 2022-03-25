# Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import argparse
import random
import socket
import sys
import time
import traceback
from collections import namedtuple
from multiprocessing import Process, Pipe
from pathlib import Path

# Prevent numpy from using up all cpu
import os
os.environ['MKL_NUM_THREADS'] = '1'  # pylint: disable=wrong-import-position

import numpy as np
import torch
import torch.optim as optim
from torch.nn.functional import smooth_l1_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from policies import DQNPolicy

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

class Logger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hostname = socket.gethostname()
        self.log_dir = Path(self.cfg.log_dir)
        print(f'log_dir: {self.log_dir}')
        self.train_summary_writer = None
        self.visualization_summary_writer = None
        self.meters = {}
        self.scalars = {}
        self.images = {}

    def update(self, name, val, add_hostname=False):
        if add_hostname:
            name = self._add_hostname(name)
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(val)

    def scalar(self, name, val, add_hostname=False):
        if add_hostname:
            name = self._add_hostname(name)
        assert name not in self.scalars
        self.scalars[name] = val

    def image(self, name, val):
        assert name not in self.images
        self.images[name] = val

    def reset(self):
        for name, meter in self.meters.items():
            assert isinstance(meter.val, (int, float)), name
            assert isinstance(meter.sum, (int, float)), name
            meter.reset()
        self.scalars = {}
        self.images = {}

    def flush(self, timestep):
        self._lazy_load_summary_writers()
        for name, meter in self.meters.items():
            self.train_summary_writer.add_scalar(name, meter.avg, timestep)
        for name, val in self.scalars.items():
            self.train_summary_writer.add_scalar(name, val, timestep)
        for name, val in self.images.items():
            self.visualization_summary_writer.add_image(name, val, timestep)
        self.reset()

    def _add_hostname(self, name):
        return f'{name}/{self.hostname}'

    def _lazy_load_summary_writers(self):
        if self.train_summary_writer is None:
            self.train_summary_writer = SummaryWriter(log_dir=str(self.log_dir / 'train'))
            self.visualization_summary_writer = SummaryWriter(log_dir=str(self.log_dir / 'visualization'))

class CollectWorker(Process):
    def __init__(self, cfg, worker_index=0, conn=None):
        super().__init__()
        self.cfg = cfg
        self.worker_index = worker_index
        self.conn = conn
        self.state = None
        self.transition_tracker = None

        if conn is None:
            self._setup()

    def _setup(self):
        # Create environment
        kwargs = {}
        self.env = utils.get_env_from_cfg(self.cfg, **kwargs)
        self.num_robot_groups = len(self.env.robot_group_types)

        self.state = self.env.reset()
        self.transition_tracker = TransitionTracker(self.state)

    def run(self):
        try:
            self._setup()
            self.conn.send(([], False, None))  # transitions_per_buffer, done, logging_info
            while True:
                self.conn.send(self.state)
                action = self.conn.recv()
                if action == 'close':
                    self.close()
                    break
                self.conn.send(self.step(action))
        except Exception as e:
            tb = traceback.format_exc()
            self.conn.send((e, tb))

    def get_state(self):
        return self.state

    def step(self, action):
        self.transition_tracker.update_action(action)
        self.state, reward, done, info = self.env.step(action)
        transitions_per_buffer = self.transition_tracker.update_step_completed(reward, self.state, done)

        logging_info = None
        if done:
            # Logging
            logging_info = {'scalars': {}, 'images': {}}
            for name in ['steps', 'simulation_steps', 'total_cubes', 'total_robot_collisions']:
                logging_info['scalars'][f'total/{name}'] = info[name]
            for i in range(self.num_robot_groups):
                for name in ['cumulative_cubes', 'cumulative_distance', 'cumulative_reward', 'cumulative_robot_collisions']:
                    logging_info['scalars'][f'cumulative/{name}/robot_group_{i + 1:02}'] = np.mean(info[name][i])

            # Reset env
            self.state = self.env.reset()
            self.transition_tracker = TransitionTracker(self.state)

        return transitions_per_buffer, done, logging_info

    def close(self):
        self.env.close()

class Collector:
    def __init__(self, cfg, policy, logger, num_workers=None):
        self.cfg = cfg
        self.policy = policy
        self.logger = logger
        self.num_workers = num_workers

        if self.num_workers is not None:
            self.curr_worker_index = 0
            self.workers = []
            self.conns = []
            for i in range(num_workers):
                parent_conn, child_conn = Pipe()
                worker = CollectWorker(self.cfg, worker_index=i, conn=child_conn)
                worker.daemon = True  # Terminate worker if parent ends
                worker.start()
                self.workers.append(worker)
                self.conns.append(parent_conn)
            self._step_fn = self._step_multiprocess
        else:
            self.worker = CollectWorker(self.cfg)
            self._step_fn = self._step

    def step(self, exploration_eps):
        collect_start_time = time.time()
        transitions_per_buffer, done, logging_info = self._step_fn(exploration_eps)

        # Logging
        if done:
            for name, val in logging_info['scalars'].items():
                self.logger.scalar(name, val)
            for name, val in logging_info['images'].items():
                self.logger.image(name, val)

        collect_time = time.time() - collect_start_time
        self.logger.update('timing/collect_time', collect_time, add_hostname=True)

        return transitions_per_buffer, done

    def _step(self, exploration_eps):
        state = self.worker.get_state()
        action = self.policy.step(state, exploration_eps=exploration_eps)
        return self.worker.step(action)

    def _step_multiprocess(self, exploration_eps):
        step_result = self.conns[self.curr_worker_index].recv()
        if isinstance(step_result[0], Exception):
            e, tb = step_result
            raise e from Exception(tb)
        transitions_per_buffer, done, logging_info = step_result
        state = self.conns[self.curr_worker_index].recv()
        action = self.policy.step(state, exploration_eps=exploration_eps)
        self.conns[self.curr_worker_index].send(action)
        self.curr_worker_index = (self.curr_worker_index + 1) % self.num_workers
        return transitions_per_buffer, done, logging_info

    def close(self):
        if self.num_workers is None:
            self.worker.close()
        else:
            for conn in self.conns:
                conn.recv()
                conn.recv()
                conn.send('close')
            for worker in self.workers:
                worker.join()

class Trainer:
    def __init__(self, cfg, policy, logger):
        self.cfg = cfg
        self.policy = policy
        self.logger = logger
        self.num_robot_groups = self.policy.num_robot_groups
        self.step_time_meter = AverageMeter()

        # Set up checkpointing
        self.checkpoint_dir = Path(self.cfg.checkpoint_dir)
        print(f'checkpoint_dir: {self.checkpoint_dir}')

        # Replay buffers
        self.replay_buffers = [ReplayBuffer(self.cfg.replay_buffer_size) for _ in range(self.num_robot_groups)]

        # Optimizers
        self.optimizers = []
        for i in range(self.num_robot_groups):
            self.optimizers.append(optim.SGD(self.policy.policy_nets[i].parameters(), lr=self.cfg.learning_rate, momentum=0.9, weight_decay=self.cfg.weight_decay))

        # Target nets
        self.target_nets = self.policy.build_policy_nets()

    def setup(self):
        start_timestep = 0
        num_episodes = 0

        # Resume if applicable
        if self.cfg.checkpoint_path is not None:
            checkpoint = torch.load(self.cfg.checkpoint_path)
            start_timestep = checkpoint['timestep']
            num_episodes = checkpoint['episodes']
            for i in range(self.num_robot_groups):
                self.optimizers[i].load_state_dict(checkpoint['optimizers'][i])
                self.replay_buffers[i] = checkpoint['replay_buffers'][i]
            print(f"=> loaded checkpoint '{self.cfg.checkpoint_path}' (timestep {start_timestep})")

        # Set up target nets
        for i in range(self.num_robot_groups):
            self.target_nets[i].load_state_dict(self.policy.policy_nets[i].state_dict())
            self.target_nets[i].eval()

        return start_timestep, num_episodes

    def _train(self, policy_net, target_net, optimizer, batch, transform_fn, discount_factor):
        state_batch = torch.cat([transform_fn(s) for s in batch.state]).to(device)  # (32, 4, 96, 96)
        action_batch = torch.tensor(batch.action, dtype=torch.long).to(device)  # (32,)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)  # (32,)
        non_final_next_states = torch.cat([transform_fn(s) for s in batch.next_state if s is not None]).to(device, non_blocking=True)  # (<=32, 4, 96, 96)

        output = policy_net(state_batch)  # (32, 2, 96, 96)
        state_action_values = output.view(self.cfg.batch_size, -1).gather(1, action_batch.unsqueeze(1)).squeeze(1)  # (32,)
        next_state_values = torch.zeros(self.cfg.batch_size, dtype=torch.float32, device=device)  # (32,)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool, device=device)  # (32,)

        if self.cfg.use_double_dqn:
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
        if self.cfg.grad_norm_clipping is not None:
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), self.cfg.grad_norm_clipping)
        optimizer.step()

        train_info = {}
        train_info['td_error'] = td_error.mean().item()
        train_info['loss'] = loss.item()

        return train_info

    def store_transitions(self, transitions_per_buffer):
        for i, transitions in enumerate(transitions_per_buffer):
            for transition in transitions:
                self.replay_buffers[i].push(*transition)

    def step(self):
        train_start_time = time.time()
        all_train_info = {}
        for i in range(self.num_robot_groups):
            assert len(self.replay_buffers[i]) >= self.cfg.batch_size
            batch = self.replay_buffers[i].sample(self.cfg.batch_size)
            train_info = self._train(self.policy.policy_nets[i], self.target_nets[i], self.optimizers[i], batch, self.policy.apply_transform, self.cfg.discount_factors[i])
            for name, val in train_info.items():
                all_train_info[f'train/{name}/robot_group_{i + 1:02}'] = val
        train_time = time.time() - train_start_time
        self.logger.update('timing/train_time', train_time, add_hostname=True)
        for name, val in all_train_info.items():
            self.logger.update(name, val)

    def update_target_networks(self):
        for i in range(self.num_robot_groups):
            self.target_nets[i].load_state_dict(self.policy.policy_nets[i].state_dict())

    def write_logs(self):
        # Visualize Q-network outputs
        assert all(len(self.replay_buffers[i]) > 0 for i in range(self.num_robot_groups))
        random_state = [[random.choice(self.replay_buffers[i].buffer).state] for i in range(self.num_robot_groups)]
        _, info = self.policy.step(random_state, debug=True)
        for i in range(self.num_robot_groups):
            visualization = utils.get_state_output_visualization(random_state[i][0], info['output'][i][0])
            self.logger.image(f'q_network/robot_group_{i + 1:02}', visualization.transpose((2, 0, 1)))

    def save_checkpoint(self, timestep, num_episodes):
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save policy
        policy_filename = f'policy_{timestep:08d}.pth.tar'
        policy_path = self.checkpoint_dir / policy_filename
        policy_checkpoint = {
            'timestep': timestep,
            'state_dicts': [self.policy.policy_nets[i].state_dict() for i in range(self.num_robot_groups)],
        }
        torch.save(policy_checkpoint, str(policy_path))

        # Save checkpoint
        checkpoint_filename = f'checkpoint_{timestep:08d}.pth.tar'
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        checkpoint = {
            'timestep': timestep,
            'episodes': num_episodes,
            'optimizers': [self.optimizers[i].state_dict() for i in range(self.num_robot_groups)],
            'replay_buffers': [self.replay_buffers[i] for i in range(self.num_robot_groups)],
        }
        torch.save(checkpoint, str(checkpoint_path))

        # Save updated config file
        self.cfg.policy_path = str(policy_path)
        self.cfg.checkpoint_path = str(checkpoint_path)
        utils.save_config(self.logger.log_dir / 'config.yml', self.cfg)

        # Remove old checkpoint
        checkpoint_paths = list(self.checkpoint_dir.glob('checkpoint_*.pth.tar'))
        checkpoint_paths.remove(checkpoint_path)
        for old_checkpoint_path in checkpoint_paths:
            old_checkpoint_path.unlink()

def main(cfg):
    # Not implemented in multiprocess training
    assert not cfg.use_predicted_intention

    # Set default multiprocess args if not specified
    if 'checkpoint_freq_mins' not in cfg:
        cfg.checkpoint_freq_mins = 30  # Checkpoint every 30 mins
    if 'num_parallel_collectors' not in cfg:
        if sys.platform == 'darwin':
            cfg.num_parallel_collectors = None
        else:
            cfg.num_parallel_collectors = 8  # 8 collect workers
    if cfg.use_egl_renderer:
        cfg.use_egl_renderer = False  # Disable EGL rendering since multiprocessing is much faster

    policy = DQNPolicy(cfg, train=True)
    logger = Logger(cfg)
    collector = Collector(cfg, policy, logger, num_workers=cfg.num_parallel_collectors)
    trainer = Trainer(cfg, policy, logger)

    # Set up trainer
    start_timestep, num_episodes = trainer.setup()
    last_checkpoint_time = -(time.time() + 60 * random.random() * cfg.checkpoint_freq_mins)

    learning_starts = round(cfg.learning_starts_frac * cfg.total_timesteps)
    total_timesteps_with_warm_up = learning_starts + cfg.total_timesteps
    for timestep in tqdm(range(start_timestep, total_timesteps_with_warm_up), initial=start_timestep, total=total_timesteps_with_warm_up, file=sys.stdout):

        step_start_time = time.time()

        # Run one collect step
        exploration_eps = 1 - (1 - cfg.final_exploration) * min(1, max(0, timestep - learning_starts) / (cfg.exploration_frac * cfg.total_timesteps))
        transitions_per_buffer, done = collector.step(exploration_eps)

        # Store transitions
        trainer.store_transitions(transitions_per_buffer)

        # Train networks
        if timestep >= learning_starts and (timestep + 1) % cfg.train_freq == 0:
            trainer.step()

        # Update target networks
        if (timestep + 1) % cfg.target_update_freq == 0:
            trainer.update_target_networks()

        # Logging
        if done:
            if timestep >= learning_starts:
                trainer.write_logs()
            num_episodes += 1
            logger.scalar('train/episodes', num_episodes)
            logger.scalar('train/exploration_eps', exploration_eps)
            logger.scalar('timing/eta', trainer.step_time_meter.avg * (total_timesteps_with_warm_up - timestep) / 3600, add_hostname=True)
            logger.flush(timestep + 1)

        # Save checkpoints
        save_checkpoint = False
        if (timestep + 1) % cfg.checkpoint_freq == 0:
            if last_checkpoint_time < 0:
                if time.time() + last_checkpoint_time > 0:
                    save_checkpoint = True
            elif time.time() - last_checkpoint_time > 60 * cfg.checkpoint_freq_mins:
                save_checkpoint = True
        if timestep + 1 == total_timesteps_with_warm_up:
            save_checkpoint = True
        if save_checkpoint:
            trainer.save_checkpoint(timestep + 1, num_episodes)
            last_checkpoint_time = time.time()

        # Log step time
        step_time = time.time() - step_start_time
        trainer.step_time_meter.update(step_time)

    # Shut down environments
    collector.close()

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
