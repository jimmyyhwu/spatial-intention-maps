# spatial-intention-maps

This code release accompanies the following paper:

### Spatial Intention Maps for Multi-Agent Mobile Manipulation

Jimmy Wu, Xingyuan Sun, Andy Zeng, Shuran Song, Szymon Rusinkiewicz, Thomas Funkhouser

*IEEE International Conference on Robotics and Automation (ICRA), 2021*

[Project Page](https://spatial-intention-maps.cs.princeton.edu) | [PDF](https://spatial-intention-maps.cs.princeton.edu/paper.pdf) | [arXiv](https://arxiv.org/abs/2103.12710) | [Video](https://youtu.be/sMTg9T1nWs4)

**Abstract:** The ability to communicate intention enables decentralized multi-agent robots to collaborate while performing physical tasks. In this work, we present spatial intention maps, a new intention representation for multi-agent vision-based deep reinforcement learning that improves coordination between decentralized mobile manipulators. In this representation, each agent's intention is provided to other agents, and rendered into an overhead 2D map aligned with visual observations. This synergizes with the recently proposed spatial action maps framework, in which state and action representations are spatially aligned, providing inductive biases that encourage emergent cooperative behaviors requiring spatial coordination, such as passing objects to each other or avoiding collisions. Experiments across a variety of multi-agent environments, including heterogeneous robot teams with different abilities (lifting, pushing, or throwing), show that incorporating spatial intention maps improves performance for different mobile manipulation tasks while significantly enhancing cooperative behaviors.

![](https://user-images.githubusercontent.com/6546428/111895195-42af8700-89ce-11eb-876c-5f98f6b31c96.gif) | ![](https://user-images.githubusercontent.com/6546428/111895197-43e0b400-89ce-11eb-953b-1ef16ac6235d.gif) | ![](https://user-images.githubusercontent.com/6546428/111895198-43e0b400-89ce-11eb-9637-84c1341228eb.gif)
:---: | :---: | :---:
![](https://user-images.githubusercontent.com/6546428/111895194-417e5a00-89ce-11eb-9f2a-1b09f96bacc8.gif) | ![](https://user-images.githubusercontent.com/6546428/111895196-43481d80-89ce-11eb-8872-0a7af6cf0bda.gif) | ![](https://user-images.githubusercontent.com/6546428/111895190-3dead300-89ce-11eb-91f4-79e056154d78.gif)

## Installation

We recommend using a [`conda`](https://docs.conda.io/en/latest/miniconda.html) environment for this codebase. The following commands will set up a new conda environment with the correct requirements (tested on Ubuntu 18.04.3 LTS):

```bash
# Create and activate new conda env
conda create -y -n my-conda-env python=3.7.10
conda activate my-conda-env

# Install mkl numpy
conda install -y numpy==1.19.2

# Install pytorch
conda install -y pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

# Install pip requirements
pip install -r requirements.txt

# Install shortest paths module (used in simulation environment)
cd shortest_paths
python setup.py build_ext --inplace
```

## Quickstart

We provide pretrained policies for each test environment. The `download-pretrained.sh` script will download the pretrained policies and save their configs and network weights into the `logs` and `checkpoints` directories, respectively. Use the following command to run it:

```bash
./download-pretrained.sh
```

You can then use `enjoy.py` to run a pretrained policy in the simulation environment. Here are a few examples you can try:

```bash
# 4 lifting robots
python enjoy.py --config-path logs/20201217T171233203789-lifting_4-small_divider-ours/config.yml
python enjoy.py --config-path logs/20201214T092812731965-lifting_4-large_empty-ours/config.yml

# 4 pushing robots
python enjoy.py --config-path logs/20201214T092814688334-pushing_4-small_divider-ours/config.yml
python enjoy.py --config-path logs/20201217T171253620771-pushing_4-large_empty-ours/config.yml

# 2 lifting + 2 pushing
python enjoy.py --config-path logs/20201214T092812868257-lifting_2_pushing_2-large_empty-ours/config.yml

# 2 lifting + 2 throwing
python enjoy.py --config-path logs/20201217T171253796927-lifting_2_throwing_2-large_empty-ours/config.yml

# 4 rescue robots
python enjoy.py --config-path logs/20210120T031916058932-rescue_4-small_empty-ours/config.yml
```

You should see the pretrained policy running in the PyBullet GUI that pops up. Here are a few examples of what it looks like (4x speed):

![](https://user-images.githubusercontent.com/6546428/111895630-3842bc80-89d1-11eb-9150-1364f80e3a26.gif) | ![](https://user-images.githubusercontent.com/6546428/111895627-35e06280-89d1-11eb-9cf7-0de0595ae68f.gif) | ![](https://user-images.githubusercontent.com/6546428/111895633-38db5300-89d1-11eb-9993-d508e6c32e7c.gif)
:---: | :---: | :---:
`lifting_4-small_divider` | `lifting_2_pushing_2-large_empty` | `rescue_4-small_empty`

You can also run `enjoy.py` without specifying a config path, and it will list all policies in the `logs` directory and allow you to pick one to run:

```bash
python enjoy.py
```

---

While the focus of this work is on multi-agent, the code also supports single-agent training. We provide a few pretrained single-agent policies which can be downloaded with the following command:

```bash
./download-pretrained.sh --single-agent
```

Here are a few example pretrained single-agent policies you can try:

```bash
# 1 lifting robot
python enjoy.py --config-path logs/20201217T171254022070-lifting_1-small_empty-base/config.yml

# 1 pushing robot
python enjoy.py --config-path logs/20201214T092813073846-pushing_1-small_empty-base/config.yml

# 1 rescue robot
python enjoy.py --config-path logs/20210119T200131797089-rescue_1-small_empty-base/config.yml
```

Here is what those policies look like when running in the PyBullet GUI (2x speed):

![](https://user-images.githubusercontent.com/6546428/111895625-34169f00-89d1-11eb-8687-689122e6b3f2.gif) | ![](https://user-images.githubusercontent.com/6546428/111895631-38db5300-89d1-11eb-9ad4-81be3908f383.gif) | ![](https://user-images.githubusercontent.com/6546428/111895632-38db5300-89d1-11eb-800e-8652d163ff1b.gif)
:---: | :---: | :---:
`lifting_1-small_empty` | `pushing_1-small_empty` | `rescue_1-small_empty`

## Training in the Simulation Environment

The [`config/experiments`](config/experiments) directory contains the template config files used for all experiments in the paper. To start a training run, you can provide one of the template config files to the `train.py` script. For example, the following will train a policy on the `SmallDivider` environment:

```bash
python train.py --config-path config/experiments/ours/lifting_4-small_divider-ours.yml
```

The training script will create a log directory and checkpoint directory for the new training run inside `logs/` and `checkpoints/`, respectively. Inside the log directory, it will also create a new config file called `config.yml`, which stores training run config variables and can be used to resume training or to load a trained policy for evaluation.

For faster training, you can try `train_multiprocess.py`, which will run 8 environments in parallel to speed up training. For example:

```bash
python train_multiprocess.py --config-path config/experiments/ours/lifting_4-small_divider-ours.yml
```

### Simulation Environment

To interactively explore the simulation environment using our dense action space (spatial action maps), you can use `tools_simple_gui.py`, which will load an environment and allow you to click on the agent's local overhead map to select navigational endpoints (each pixel is an action). Some robot types (such as lifting) have a 2-channel action space, in which case you would use left click to move, and right click to move and then attempt an end effector action at the destination (such as lift or throw).

```bash
python tools_simple_gui.py
```

Note that `tools_simple_gui.py` currently only supports single-agent environments. For multi-agent environments, you can use `tools_interactive_gui.py` which has many more features, including control of multiple agents:

```bash
python tools_interactive_gui.py
```

### Evaluation

Trained policies can be evaluated using the `evaluate.py` script, which takes in the config path for the training run. For example, to evaluate the `SmallDivider` pretrained policy, you can run:

```
python evaluate.py --config-path logs/20201217T171233203789-lifting_4-small_divider-ours/config.yml
```

This will load the trained policy from the specified training run, and run evaluation on it. The results are saved to an `.npy` file in the `eval` directory. You can then run `jupyter notebook` and navigate to [`eval_summary.ipynb`](eval_summary.ipynb) to load the `.npy` files and generate tables and plots of the results.

## Running in the Real Environment

We train policies in simulation and run them directly on the real robot by mirroring the real environment inside the simulation. To do this, we first use [ArUco](https://docs.opencv.org/4.4.0/d5/dae/tutorial_aruco_detection.html) markers to estimate 2D poses of robots and objects in the real environment, and then use the estimated poses to update the simulation. Note that setting up the real environment, particularly the marker pose estimation, can take a fair amount of time and effort.

### Vector SDK Setup

If you previously ran `pip install -r requirements.txt` following the installation instructions above, the `anki_vector` library should already be installed. Run the following command to set up each robot you plan to use:

```bash
python -m anki_vector.configure
```

After the setup is complete, you can open the Vector config file located at `~/.anki_vector/sdk_config.ini` to verify that all of your robots are present.

You can also run some of the [official examples](https://developer.anki.com/vector/docs/downloads.html#sdk-examples) to verify that the setup procedure worked. For further reference, please see the [Vector SDK documentation](https://developer.anki.com/vector/docs/index.html).

### Connecting to the Vector

The following command will try to connect to all the robots in your Vector config file and keep them still. It will print out a message for each robot it successfully connects to, and can be used to verify that the Vector SDK can connect to all of your robots.

```bash
python vector_keep_still.py
```

**Note:** If you get the following error, you will need to make a small fix to the `anki_vector` library.

```
AttributeError: module 'anki_vector.connection' has no attribute 'CONTROL_PRIORITY_LEVEL'
```

Locate the `anki_vector/behavior.py` file inside your installed conda libraries. The full path should be in the error message. At the bottom of `anki_vector/behavior.py`, change `connection.CONTROL_PRIORITY_LEVEL.RESERVE_CONTROL` to `connection.ControlPriorityLevel.RESERVE_CONTROL`.

---

Sometimes the IP addresses of your robots will change. To update the Vector config file with new IP addresses, you can run the following command:

```bash
python vector_run_mdns.py
```

The script uses mDNS to find all Vector robots on the local network, and will automatically update their IP addresses in the Vector config file. It will also print out the hostname, IP address, and MAC address of every robot found. Make sure `zeroconf` is installed (`pip install zeroconf`) or mDNS may not work well. Alternatively, you can just open the Vector config file at `~/.anki_vector/sdk_config.ini` in a text editor and manually update the IP addresses.

### Controlling the Vector

The `vector_keyboard_controller.py` script is adapted from the [remote control example](https://github.com/anki/vector-python-sdk/blob/master/examples/apps/remote_control/remote_control.py) in the official SDK, and can be used to verify that you are able to control the robot using the Vector SDK. Use it as follows:

```bash
python vector_keyboard_controller.py --robot-index ROBOT_INDEX
```

The `--robot-index` argument specifies the robot you wish to control and refers to the index of the robot in the Vector config file (`~/.anki_vector/sdk_config.ini`).

### Building the Real Environment

Please reference the videos on the [project page](https://spatial-intention-maps.cs.princeton.edu) when building the real environment setup.

We built the walls using 50 cm x 44 mm strips of Elmer's Foam Board. We also use several 3D printed parts, which we printed using the [Sindoh 3DWOX 1](https://www.amazon.com/Sindoh-3DWOX-Printer-New-Model/dp/B07C79C9RB) 3D printer (with PLA filament). All 3D model files are in the [`stl`](stl) directory.

Here are the different parts to 3D print for the environment setup:
* `cube.stl`: the objects that the robots will forage
* `wall-support.stl`: triangular supports used to secure the walls to the tabletop
* `rounded-corner.stl`: rounded blocks installed in corners of the environment to allow pushing through corners
* `board-corner.stl`: used for pose estimation with ArUco markers

Additionally, a 3D printed attachment needs to be installed on each robot to enable its special abilities:
* `lifting-attachment.stl`: attach to bottom of Vector's lift, allows the lifting robot (and rescue robot) to align with objects
* `pushing-attachment.stl`: attach to front of Vector's lift, allows the pushing robot to push objects more predictably
* `throwing-attachment.stl`: attach to arms of Vector's lift, allows the throwing robot to throw objects backwards

Note that all attachments need to be secured to the robot (using tape, for example). The robots will not be able to reliably execute their end effector action with loose attachments.

There are also a few things to print in the [`printouts`](printouts) directory:
* `back-covers.pdf`: attach to back of throwing robot to make throws more consistent (recommend printing on cardstock)
* `receptacle.pdf`: the target receptacle, install in the top right corner of the room

### Running Trained Policies on the Real Robot

First see the [`aruco`](aruco) directory for instructions on setting up pose estimation with ArUco markers.

Once the setup is completed, make sure the pose estimation server is started before proceeding:

```bash
cd aruco
python server.py
```

---

We can use `tools_simple_gui.py` from before to manually control a robot in the real environment too, which will allow us to verify that all components of the real setup are working properly, including pose estimation and robot control. See the bottom of the `main` function in `tools_simple_gui.py` ([L100](tools_simple_gui.py#L100)) for the appropriate arguments. You will need to enable `real` and provide values for `real_robot_indices` and `real_cube_indices`. You can then run the same command from before to start the GUI:

```bash
python tools_simple_gui.py
```

You should see that the simulation environment in the PyBullet GUI mirrors the real setup with millimeter-level precision. If the poses in the simulation do not look correct, you can restart the pose estimation server with the `--debug` flag to enable debug visualizations:

```bash
cd aruco
python server.py --debug
```

As previously noted, `tools_simple_gui.py` currently only supports single-agent control.

For multi-agent control, you can use `tools_interactive_gui.py`. After enabling `real` and providing values for `real_robot_indices` and `real_cube_indices`, you can run:

```bash
python tools_interactive_gui.py
```

---

Once you have verified that manual control with `tools_simple_gui.py` works, you can then run a trained policy using `enjoy.py` from before. For example, to run the `SmallDivider` pretrained policy in the real environment, you can run:

```bash
python enjoy.py --config-path logs/20201217T171233203789-lifting_4-small_divider-ours/config.yml --real --real-robot-indices 0,1,2,3 --real-cube-indices 0,1,3,5,6,7,8,9,10,11
```

For debugging and visualization, `tools_interactive_gui.py` allows you to load a trained policy and interactively run the policy one step at a time while showing Q-value maps and transitions. For example:

```bash
python tools_interactive_gui.py --config-path logs/20201217T171233203789-lifting_4-small_divider-ours/config.yml
```

## Citation

If you find this work useful for your research, please consider citing:

```
@inproceedings{wu2021spatial,
  title = {Spatial Intention Maps for Multi-Agent Mobile Manipulation},
  author = {Wu, Jimmy and Sun, Xingyuan and Zeng, Andy and Song, Shuran and Rusinkiewicz, Szymon and Funkhouser, Thomas},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year = {2021}
}
```
