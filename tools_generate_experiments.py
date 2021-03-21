from pathlib import Path
import utils

def generate_experiment(experiment_name, template_experiment_name, modify_cfg_fn, output_dir, template_dir='config/experiments/base'):
    # Ensure output dir exists
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Read template config
    cfg = utils.load_config(Path(template_dir) / '{}.yml'.format(template_experiment_name))

    # Apply modifications
    cfg.experiment_name = experiment_name
    num_fields = len(cfg)
    modify_cfg_fn(cfg)
    assert num_fields == len(cfg), experiment_name  # New fields should not have been added

    # Save new config
    utils.save_config(output_dir / '{}.yml'.format(experiment_name), cfg)

def get_discount_factors(robot_config, offset=0):
    discount_factor_list = [0.2, 0.35, 0.5, 0.65, 0.75, 0.85]
    start_indices = {
        'lifting_robot': 4,
        'pushing_robot': 4,
        'throwing_robot': 4,
        'rescue_robot': 0,
    }
    num_robots = sum(next(iter(g.values())) for g in robot_config)
    robot_group_types = [next(iter(g.keys())) for g in robot_config]
    discount_factors = []
    for robot_type in robot_group_types:
        idx = start_indices[robot_type]
        if num_robots > 1:
            idx += 1
        idx += offset
        discount_factors.append(discount_factor_list[idx])
    return discount_factors

assert get_discount_factors([{'lifting_robot': 1}]) == [0.75]
assert get_discount_factors([{'pushing_robot': 1}]) == [0.75]
assert get_discount_factors([{'throwing_robot': 1}]) == [0.75]
assert get_discount_factors([{'rescue_robot': 1}]) == [0.2]
assert get_discount_factors([{'lifting_robot': 4}]) == [0.85]
assert get_discount_factors([{'pushing_robot': 4}]) == [0.85]
assert get_discount_factors([{'rescue_robot': 4}]) == [0.35]

def main():
    ################################################################################
    # Robot types

    def modify_cfg_lifting_to_lifting(cfg):
        cfg.discount_factors = get_discount_factors(cfg.robot_config)
        cfg.total_timesteps = 40000

    def modify_cfg_lifting_to_pushing(cfg):
        cfg.robot_config = [{'pushing_robot': 1}]
        cfg.discount_factors = get_discount_factors(cfg.robot_config)
        cfg.total_timesteps = 60000

    def modify_cfg_lifting_to_rescue(cfg):
        cfg.robot_config = [{'rescue_robot': 1}]
        cfg.discount_factors = get_discount_factors(cfg.robot_config)
        cfg.total_timesteps = 3750
        cfg.num_input_channels -= 1
        utils.apply_misc_env_modifications(cfg, 'rescue_1')

    output_dir = 'config/experiments/base'
    generate_experiment('lifting_1-small_empty-base', 'lifting_1-small_empty', modify_cfg_lifting_to_lifting, output_dir, template_dir='config/templates')
    generate_experiment('pushing_1-small_empty-base', 'lifting_1-small_empty', modify_cfg_lifting_to_pushing, output_dir, template_dir='config/templates')
    generate_experiment('rescue_1-small_empty-base', 'lifting_1-small_empty', modify_cfg_lifting_to_rescue, output_dir, template_dir='config/templates')

    ################################################################################
    # Multi-agent

    def modify_cfg_multi_agent(cfg, robot_config):
        cfg.robot_config = robot_config
        num_robots = sum(next(iter(g.values())) for g in cfg.robot_config)
        cfg.total_timesteps *= num_robots
        cfg.train_freq = num_robots
        cfg.discount_factors = get_discount_factors(cfg.robot_config)

    # Homogeneous
    output_dir = 'config/experiments/base'
    num_robots = 4
    for template_experiment_name in [
            'lifting_1-small_empty-base',
            'pushing_1-small_empty-base',
            'rescue_1-small_empty-base',
        ]:
        experiment_name = template_experiment_name.replace('_1', '_{}'.format(num_robots))
        if 'lifting' in template_experiment_name:
            robot_config = [{'lifting_robot': num_robots}]
        elif 'pushing' in template_experiment_name:
            robot_config = [{'pushing_robot': num_robots}]
        elif 'rescue' in template_experiment_name:
            robot_config = [{'rescue_robot': num_robots}]
        else:
            raise Exception
        generate_experiment(experiment_name, template_experiment_name, lambda x: modify_cfg_multi_agent(x, robot_config), output_dir)

    # Heterogeneous
    output_dir = 'config/experiments/base'
    template_experiment_name = 'lifting_1-small_empty-base'
    experiment_name = 'lifting_2_throwing_2-small_empty-base'
    robot_config = [{'lifting_robot': 2}, {'throwing_robot': 2}]
    generate_experiment(experiment_name, template_experiment_name, lambda x: modify_cfg_multi_agent(x, robot_config), output_dir)
    template_experiment_name = 'pushing_1-small_empty-base'
    experiment_name = 'lifting_2_pushing_2-small_empty-base'
    robot_config = [{'lifting_robot': 2}, {'pushing_robot': 2}]
    generate_experiment(experiment_name, template_experiment_name, lambda x: modify_cfg_multi_agent(x, robot_config), output_dir)

    ################################################################################
    # Config for local development

    def modify_cfg_to_local(cfg):
        cfg.logs_dir = 'logs'
        cfg.checkpoints_dir = 'checkpoints'
        cfg.batch_size = 4
        cfg.replay_buffer_size = 1000
        cfg.learning_starts_frac = 0.0000625
        cfg.inactivity_cutoff_per_robot = 5
        cfg.show_gui = True
        cfg.use_egl_renderer = False

    output_dir = 'config/local'
    for template_experiment_name in ['lifting_4-small_empty-base']:
        experiment_name = template_experiment_name.replace('base', 'local')
        generate_experiment(experiment_name, template_experiment_name, modify_cfg_to_local, output_dir)

    ################################################################################
    # Environments

    def modify_cfg_env_name(cfg, env_name):
        cfg.env_name = env_name
        utils.apply_misc_env_modifications(cfg, env_name)

    output_dir = 'config/experiments/base'

    # Lifting
    for template_experiment_name in ['lifting_1-small_empty-base', 'lifting_4-small_empty-base']:
        for env_name in ['small_divider', 'large_empty', 'large_doors', 'large_tunnels', 'large_rooms']:
            experiment_name = template_experiment_name.replace('small_empty', env_name)
            generate_experiment(experiment_name, template_experiment_name, lambda x: modify_cfg_env_name(x, env_name), output_dir)

    # Pushing
    for template_experiment_name in ['pushing_1-small_empty-base', 'pushing_4-small_empty-base']:
        for env_name in ['small_divider', 'large_empty']:
            experiment_name = template_experiment_name.replace('small_empty', env_name)
            generate_experiment(experiment_name, template_experiment_name, lambda x: modify_cfg_env_name(x, env_name), output_dir)

    # Lifting and pushing
    template_experiment_name = 'lifting_2_pushing_2-small_empty-base'
    for env_name in ['large_empty', 'large_doors', 'large_rooms']:
        experiment_name = template_experiment_name.replace('small_empty', env_name)
        generate_experiment(experiment_name, template_experiment_name, lambda x: modify_cfg_env_name(x, env_name), output_dir)

    # Lifting and throwing
    template_experiment_name = 'lifting_2_throwing_2-small_empty-base'
    for env_name in ['large_empty', 'large_doors']:
        experiment_name = template_experiment_name.replace('small_empty', env_name)
        generate_experiment(experiment_name, template_experiment_name, lambda x: modify_cfg_env_name(x, env_name), output_dir)

    # Rescue
    for template_experiment_name in ['rescue_1-small_empty-base', 'rescue_4-small_empty-base']:
        env_name = 'large_empty'
        experiment_name = template_experiment_name.replace('small_empty', env_name)
        generate_experiment(experiment_name, template_experiment_name, lambda x: modify_cfg_env_name(x, env_name), output_dir)

    ################################################################################
    # Ours (intention map with ramp encoding)

    def modify_cfg_intention_map(cfg, encoding):
        cfg.use_intention_map = True
        cfg.intention_map_encoding = encoding
        cfg.num_input_channels += 1

    for template_experiment_path in sorted(Path('config/experiments/base').glob('*.yml')):
        if template_experiment_path.name.startswith(('lifting_1', 'pushing_1', 'rescue_1', 'throwing_1')):
            continue
        if template_experiment_path.name.startswith(('lifting_2_pushing_2-small_empty', 'lifting_2_throwing_2-small_empty')):
            continue
        template_experiment_name = template_experiment_path.name.replace('.yml', '')
        experiment_name = template_experiment_name.replace('base', 'ours')
        generate_experiment(experiment_name, template_experiment_name, lambda cfg: modify_cfg_intention_map(cfg, 'ramp'), 'config/experiments/ours')

    ################################################################################
    # Comparisons and ablations

    template_experiment_names = [
        'lifting_4-large_doors-base',
        'lifting_4-large_empty-base',
        'lifting_4-large_rooms-base',
        'lifting_4-large_tunnels-base',
        'lifting_4-small_divider-base',
        'lifting_4-small_empty-base',
    ]

    def modify_cfg_use_intention_channels(cfg, encoding):
        cfg.use_intention_channels = True
        cfg.intention_channel_encoding = encoding
        num_robots = sum(sum(g.values()) for g in cfg.robot_config)
        cfg.num_input_channels += (2 if cfg.intention_channel_encoding == 'nonspatial' else 1) * (num_robots - 1)

    def modify_cfg_use_history_map(cfg):
        cfg.use_history_map = True
        cfg.num_input_channels += 1

    def modify_cfg_use_predicted_intention(cfg):
        cfg.use_predicted_intention = True
        cfg.num_input_channels += 1

    def modify_cfg_use_predicted_intention_with_history(cfg):
        modify_cfg_use_history_map(cfg)
        modify_cfg_use_predicted_intention(cfg)

    # Intention map variants
    output_dir = 'config/experiments/comparisons/intention_maps'
    for template_experiment_name in template_experiment_names:
        for variant in ['binary', 'line', 'circle']:
            experiment_name = template_experiment_name.replace('base', variant)
            generate_experiment(experiment_name, template_experiment_name, lambda cfg: modify_cfg_intention_map(cfg, variant), output_dir)

    # Intention channels
    output_dir = 'config/experiments/comparisons/intention_channels'
    for template_experiment_name in template_experiment_names:
        for encoding in ['spatial', 'nonspatial']:
            experiment_name = template_experiment_name.replace('base', encoding)
            generate_experiment(experiment_name, template_experiment_name, lambda cfg: modify_cfg_use_intention_channels(cfg, encoding), output_dir)

    # History maps
    output_dir = 'config/experiments/comparisons/history_maps'
    for template_experiment_name in template_experiment_names:
        experiment_name = template_experiment_name.replace('base', 'history')
        generate_experiment(experiment_name, template_experiment_name, modify_cfg_use_history_map, output_dir)

    # Predicted intention
    output_dir = 'config/experiments/comparisons/predicted_intention'
    for template_experiment_name in template_experiment_names:
        experiment_name = template_experiment_name.replace('base', 'predicted')
        generate_experiment(experiment_name, template_experiment_name, modify_cfg_use_predicted_intention, output_dir)
        experiment_name = template_experiment_name.replace('base', 'predicted_with_history')
        generate_experiment(experiment_name, template_experiment_name, modify_cfg_use_predicted_intention_with_history, output_dir)

main()
