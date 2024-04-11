import time
import yaml
from src.federation.server import Server
import torch
import torch.multiprocessing as mp
import argparse

from src.utils.fed_utils import make_fed_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--algorithm',
        '-alg',
        type=str,
        default='SAC',
        dest='algorithm',
        help='Algorithm used to train (possible values: SAC, TD3).')

    parser.add_argument(
        '--config',
        '-cf',
        type=str,
        dest='config_file',
        help='Path to the configuration file that specifies training context.'
    )

    parser.add_argument(
        '--seed',
        '-s',
        type=int,
        dest='seed',
        default=10,
        help='Sets seed of random number generators.'
    )

    parser.add_argument(
        '--eval',
        action='store_true',
        dest='eval',
        help='By setting this flag, the model(s) will be evaluated at the end of every n episodes. \
            The frequency and length of the evaluation is determined by the config file provided.'
    )

    parser.add_argument(
        '--mask_thres',
        '-mt',
        type=float,
        dest='masking_threshold',
        default=0.4,
        help='Masking threshold to use for gradient masking (federated algorithms)'
    )

    parser.add_argument(
        '--local_updates_per_round',
        '-lupr',
        type=int,
        dest='local_updates_per_round',
        default=4,
        help='How many local updates for each client to perform before one global update.'
    )

    parser.add_argument(
        '--global_lr',
        '-glr',
        type=float,
        dest='global_learning_rate',
        default=1.0,
        help='The learning rate used for the global update steps.'
    )

    parser.add_argument(
        '--client_lr',
        '-clr',
        type=float,
        dest='client_learning_rate',
        default=3e-4,
        help='The learning rate used for the local update steps.'
    )

    parser.add_argument(
        '--client_optimizer',
        '-copt',
        type=str,
        dest='client_optimizer',
        default='SGD',
        help='The optimizer used in the local training processes (possible values SGD, Adam). '
    )

    parser.add_argument(
        '--momentum',
        '-mom',
        type=float,
        dest='momentum',
        default=0.0,
        help='The momentum used if client uses SGD as optimizer.'
    )

    parser.add_argument(
        '--server_momentum',
        '-ser_mom',
        type=float,
        dest='server_momentum',
        default=0.9,
        help='The momentum used by server in FedAvgM algorithm.'
    )

    parser.add_argument(
        '--beta_1',
        '-b_1',
        type=float,
        dest='beta_1',
        default=0.9,
        help='The first moment parameter of the adaptive federated optimizer (applicable to FedAdam and FedYogi).'
    )

    parser.add_argument(
        '--beta_2',
        '-b_2',
        type=float,
        dest='beta_2',
        default=0.999,
        help='The second moment parameter of the adaptive federated optimizer (applicable to FedAdam and FedYogi).'
    )

    parser.add_argument(
        '--max_ep_data_store_num',
        '-medsn',
        type=int,
        dest='max_ep_data_store_num',
        default=2,
        help='How many sinergym simulation episode logs (subrun folders) to store. \
            The logs require a significant amount of memory, so it is recommended to keep \
            it low when running several experiments.'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        dest='verbose',
        help='By setting this flag, you will get a more detailed stdout of the training progress.'
    )
    args = parser.parse_args()

    # Set multiprocessing start method
    mp.set_start_method("spawn")

    # read configuration file
    with open(args.config_file) as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
    
    envs = configs[0]["environments"]
    eval_env = configs[0]["eval_env"]
    global_config = configs[1]["global_config"]
    global_config["seed"] = args.seed
    global_config["max_ep_data_store_num"] = args.max_ep_data_store_num
    wrapper_config = configs[2]["wrapper_config"]
    init_kwargs = configs[3]["init_kwargs"]
    fed_config = configs[5]["fed_config"]
    eval_config = configs[6]["eval_config"]
    eval_config["eval"] = args.eval

    init_kwargs['learning_rate'] = args.client_learning_rate
    if configs[4]["init_policy_kwargs"] is None:
        configs[4]["init_policy_kwargs"] = {}
    init_policy_kwargs = configs[4]["init_policy_kwargs"]

    if args.client_optimizer.lower() == 'sgd':
        init_policy_kwargs["optimizer_class"] = torch.optim.SGD
        init_policy_kwargs["optimizer_kwargs"] = {'momentum': args.momentum}
    elif args.client_optimizer.lower() == 'adam':
        init_policy_kwargs["optimizer_class"] = torch.optim.Adam
    else:
        raise RuntimeError("Invalid client optimizer requested! Must be either SGD or Adam.")

    if fed_config is not None:
        fed_config = make_fed_config(fed_config, args)
    
    # display experiment configuration
    message = "\n[WELCOME] Unfolding configurations...!"
    print(message)
    print()

    for config in configs:
        print(config)
        print()
    
    train_federated = True
    if fed_config is None:
        train_federated = False

    # initialize Server 
    central_server = Server(algo=args.algorithm,
                            envs=envs,
                            eval_env=eval_env,
                            global_config=global_config,
                            wrapper_config=wrapper_config,
                            init_kwargs=init_kwargs,
                            init_policy_kwargs=init_policy_kwargs,
                            fed_config=fed_config,
                            eval_config=eval_config,
                            train_federated=train_federated,
                            verbose=args.verbose)
   
    # Perform server setup
    central_server.setup()

    # Execute entire training process
    central_server.train()
    
    # End of training process
    message = "...done all learning process!\n...exit program!"
    print(message, flush=True)
    time.sleep(3); exit()