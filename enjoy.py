import argparse
import utils

def main(args):
    config_path = args.config_path
    if config_path is None:
        config_path = utils.select_run()
    if config_path is None:
        return
    print(config_path)
    cfg = utils.load_config(config_path)

    # Create env
    if args.real:
        real_robot_indices = list(map(int, args.real_robot_indices.split(',')))
        real_cube_indices = list(map(int, args.real_cube_indices.split(',')))
        env = utils.get_env_from_cfg(cfg, real=True, real_robot_indices=real_robot_indices, real_cube_indices=real_cube_indices)
    else:
        env = utils.get_env_from_cfg(cfg, show_gui=True)

    # Create policy
    policy = utils.get_policy_from_cfg(cfg)

    # Run policy
    state = env.reset()
    try:
        while True:
            action = policy.step(state)
            state, _, done, _ = env.step(action)
            if done:
                state = env.reset()
    finally:
        env.close()

parser = argparse.ArgumentParser()
parser.add_argument('--config-path')
parser.add_argument('--real', action='store_true')
parser.add_argument('--real-robot-indices', default='0,1,2,3')
parser.add_argument('--real-cube-indices', default='0,1,3,5,6,7,8,9,10,11')
main(parser.parse_args())
