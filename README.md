## Table of contents

* [Introduction](#introduction)
* [Requirements](#requirements)
* [How to use](#how-to-use)
* [Resources](#resources)
* [Reference](#reference)
* [License](#license)

## Introduction

This repository contains an implementation of the double deep Q-learning (DDQN) approach to control multiple UAVs on a data harvesting from IoT sensors mission, including dual global-local map processing. The corresponding paper ["Multi-UAV Path Planning for Wireless Data Harvesting with Deep Reinforcement Learning"](https://arxiv.org/abs/2010.12461) is available on arXiv.

An earlier single-UAV conference version ["UAV Path Planning for Wireless Data Harvesting: A Deep Reinforcement Learning Approach"](https://arxiv.org/abs/2007.00544) is also available on arXiv and was presented at IEEE Globecom 2020.

For questions, please contact [Harald Bayerlein](https://hbay.gitlab.io) via email harald.bayerlein@eurecom.fr. Please also note that due to github's new naming convention, the 'master' branch is now called 'main' branch.


## Requirements

```
python==3.7 or newer
numpy==1.18.5 or newer
keras==2.4.3 or newer
tensorflow==2.3.0 or newer
matplotlib==3.3.0 or newer
scikit-image==0.16.2 or newer
tqdm==4.45.0 or newer
```


## How to use

Train a new DDQN model with the parameters of your choice in the specified config file, e.g. with the standard config for the 'manhattan32' map:

```
python main.py --gpu --config config/manhattan32.json --id manhattan32

--gpu                       Activates GPU acceleration for DDQN training
--config                    Path to config file in json format
--id                        Overrides standard name for logfiles and model
--generate_config           Enable only to write default config from default values in the code
```

For keeping track of the training, use TensorBoard. Various performance and training metrics, as well as intermittent test plots of trajectories, are recorded in log files and automatically saved in the 'logs' directory. On the command line, run:

```
tensorboard --logdir logs
```

Evaluate a model (saved during training in the 'models' directory) through Monte Carlo analysis over the random parameter space for the performance indicators 'Successful Landing', 'Collection Ratio', 'Collection Ratio and Landed' as defined in the paper (plus 'Boundary Counter' counting safety controller activations), e.g. for 1000 Monte Carlo iterations:

```
python main_mc.py --weights models/manhattan32_best --config config/manhattan32.json --id manhattan32_mc --samples 1000

--weights                   Path to weights of trained model
--config                    Path to config file in json format
--id                        Name for exported files
--samples                   Number of Monte Carlo  over random scenario parameters
--seed                      Seed for repeatability
--show                      Pass '--show True' for individual plots of scenarios and allow plot saving
--num_agents                Overrides number of agents range, e.g. 12 for random range of [1,2] agents, or 11 for single agent
```

With the most recent code update, the config options and associated code to train agents with either 'scalar' input (that means no map, but only concatenated numerical values as state information as described in [1]) or 'blind' (only ego UAV position and remaining flying time as state input) were added. These are only relevant for the comparison in paper [1] and can be savely ignored when you are only interested in the map-based state input.

## Resources

The city environments from the paper 'manhattan32' and 'urban50' are included in the 'res' directory. Map information is formatted as PNG files with one pixel representing on grid world cell. The pixel color determines the type of cell according to

* red #ff0000 no-fly zone (NFZ)
* green #00ff00 buildings blocking wireless links (UAVs can fly over)
* blue #0000ff start and landing zone
* yellow #ffff00 buildings blocking wireless links + NFZ (UAVs can not fly over)

If you would like to create a new map, you can use any tool to design a PNG with the same pixel dimensions as the desired map and the above color codes.

The shadowing maps, defining for each position and each IoT device whether there is a line-of-sight (LoS) or non-line-of-sight (NLoS) connection, are computed automatically the first time a new map is used for training and then saved to the 'res' directory as an NPY file.


## Reference

If using this code for research purposes, please cite:

[1] H. Bayerlein, M. Theile, M. Caccamo, and D. Gesbert, “Multi-UAV path planning for wireless data harvesting with deep reinforcement learning," arXiv:2010.12461 [cs.MA], 2020. 

```
@article{Bayerlein2020,
        author  = {Harald Bayerlein and Mirco Theile and Marco Caccamo and David Gesbert},
        title   = {Multi-{UAV} Path Planning for Wireless Data Harvesting with Deep Reinforcement Learning},
        journal = {arXiv:2010.12461 [cs.MA]},
        year    = {2020},
        url     = {https://arxiv.org/abs/2010.12461}
}
```

Or for the shorter single-UAV conference version:

[2] H. Bayerlein, M. Theile, M. Caccamo, and D. Gesbert, “UAV path planning for wireless data harvesting: A deep reinforcement learning approach,” in IEEE Global Communications Conference (GLOBECOM), 2020.

```
@inproceedings{Bayerlein2020short,
  author={Harald Bayerlein and Mirco Theile and Marco Caccamo and David Gesbert},
  title={{UAV} Path Planning for Wireless Data Harvesting: A Deep Reinforcement Learning Approach}, 
  booktitle={IEEE Global Communications Conference (GLOBECOM)}, 
  year={2020}
}
```


## License 

This code is under a BSD license.