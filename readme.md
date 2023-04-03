# Evolutionary Multi-Objective Deep Reinforcement Learning for Autonomous UAV Navigation in Large-Scale Complex Environments

The implementation of the proposed framework in [Evolutionary Multi-Objective Deep Reinforcement Learning for Autonomous UAV Navigation in Large-Scale Complex Environments](https://doi.org/10.1145/3583131.3590446).

## Notes
This program requires a long time to run, with the current settings,
on a server with 8x2080GPU and Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz, 32 thread,
it takes about 3.5 days to run. Each thread takes about 1GB GPU memory.

## Setup

The conda environment file is [env.yaml](./env.yaml).
Here we use PyTorch with CUDA 2.0.0, which might not work with the latest GPUs.
Update to newer version of PyTorch using instructions from [PyTorch Install Guide](https://pytorch.org/get-started/locally/)

crete new environment by 

```shell
 conda env create -n <ENVNAME> --file env.yaml  # with pytorch 2.0.0 and cuda 11.7
```

or install the dependencies manually:

```
# REQUIRED
pytorch  # should work with any version >= 1.0.0, but only tested on 1.11, 1.13, 2.0.
gym==0.21.0  # gym has breaking changes in 0.22.0
pymoo==0.5.0  # pymoo has breaking changes in 0.6.0

# OPTIONAL
pygame  # for visualization, if not installed, you don't need to do anything with the code
```

**REQUIRED** to compile the C++ extension (to speed up execution).

```shell
# requires cmake and g++
# could be installed by:
# sudo apt install build-essential cmake
# or on Windows, install Visual Studio and CMake
# otherwise, you can just follow the procedure in the build file and compile it manually

cd gym_uav_c_functions
./build.sh
# on Windows
# build.bat
```

Before running, you *might* want to edit the configs in the [main.py](./main.py),
which are marked with `# TODO: <what is it>`.

Run the experiment by:

```shell
python main.py
```

or run in background with:

```shell
nohup python ./main.py > out.txt 2>&1 &
```

## Structure
* [ddpg_pytorch](./ddpg_pytorch) dir contains the implementation of DDPG with PyTorch, see the LICENSE file for more info.
* [gym_uav](./gym_uav) dir contains the implementation of the (gym) UAV environment.
* [gym_uav_c_functions](./gym_uav_c_functions) dir contains the C++ extension for the UAV environment.
* [uav_problem](./uav_problem) dir contains the implementation of the (pymoo) UAV MOP.
* [main.py](./main.py) is the main file to run the experiment.
