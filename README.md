# FedHVAC

This repo is the codebase used in the research article ["Employing Federated Learning for Training Autonomous HVAC Systems"](https://arxiv.org/abs/2405.00389). 

Instructions for running code COMING SOON.

## Abstract

Buildings account for 40 \% of global energy consumption. A considerable portion of building energy consumption stems from heating, ventilation, and air conditioning (HVAC), and thus implementing smart, energy-efficient HVAC systems has the potential to significantly impact the course of climate change. In recent years, model-free reinforcement learning algorithms have been increasingly assessed for this purpose due to their ability to learn and adapt purely from experience. They have been shown to outperform classical controllers in terms of energy cost and consumption, as well as thermal comfort. However, their weakness lies in their relatively poor data efficiency, requiring long periods of training to reach acceptable policies, making them inapplicable to real-world controllers directly. Hence, common research goals are to improve the learning speed, as well as to improve their ability to generalize, in order to facilitate transfer learning to unseen building environments.

In this paper, we take a federated learning approach to training the reinforcement learning controller of an HVAC system. A global control policy is learned by aggregating local policies trained on multiple data centers located in different climate zones. The goal of the policy is to simultaneously minimize energy consumption and maximize thermal comfort. The federated optimization strategy indirectly increases both the rate at which experience data is collected and the variation in the data. We demonstrate through experimental evaluation that these effects lead to a faster learning speed, as well as greater generalization capabilities in the federated policy compared to any individually trained policy. Furthermore, the learning stability is significantly improved, with the learning process and performance of the federated policy being less sensitive to the choice of parameters and the inherent randomness of reinforcement learning. We perform a thorough set of experiments, evaluating three different optimizers for local policy training, as well as three different federated learning algorithms.

## Installation

The *FedHVAC* software has only been tested with [Ubuntu](https://ubuntu.com/) 20.04, [Sinergym](https://github.com/ugr-sail/sinergym/tree/main) v2.0.0, 
[EnergyPlus](https://energyplus.net) v9.5.0 and [BCVTB](https://simulationresearch.lbl.gov/bcvtb/) v1.6.0. We cannot guarantee that
the software will work using any other versions of the mentioned software. 

To install *FedHVAC*, perform the following steps:

#### 1. Clone the repository

```sh
    $ git clone https://github.com/hagstromf/FedHVAC.git
```

#### 2. Configure Conda environment

We have provided the file `fed_hvac_environment.yml` to install all the necessary libraries to run *FedHVAC* using conda. 
To configure the conda environment, run the following commands:

```sh
    $ cd FedHVAC
    $ conda env create -f fed_hvac_environment.yml
    $ conda activate fed_hvac
```

#### 3. Install EnergyPlus

Install EnergyPlus version 9.5.0. Follow the instructions [here](https://energyplus.net/downloads) and
install it for Linux (only Ubuntu 20.04 is supported). Choose any location to install the software. 
Once installed, a folder called `Energyplus-9-5-0` should appear in the selected location.

#### 4. Install BCVTB 

Follow the instructions [here](https://simulationresearch.lbl.gov/bcvtb/Download) for
installing BCVTB (v1.6.0) software. Choose any location to install the software. 
Another option is to copy the `bcvtb` folder from [this repository](https://github.com/zhangzhizza/Gym-Eplus/tree/master/eplus_env/envs).

#### 5. Set environment variables

Two environment variables must be set: `EPLUS_PATH` and
`BCVTB_PATH`, with the locations where EnergyPlus and BCVTB are
installed respectively.

```sh
    $ export EPLUS_PATH=PATH/TO/Energyplus-9-5-0
    $ export BCVTB_PATH=PATH/TO/bcvtb
```
