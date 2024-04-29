# FedHVAC

This repo is the codebase used in the research article "Employing Federated Learning for Training Autonomous HVAC Systems" which has been submitted for review. 
Link will be added once paper has been published.

Instructions for running code COMING SOON.

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