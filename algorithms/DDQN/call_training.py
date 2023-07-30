import logger
from argparse import ArgumentParser
import torch as T
import hockey_env as env
from agent import DQNAgent
from training import Trainer
import os
import time

## TODO adapt and try out default values
arg_parser = ArgumentParser()
# Arguments for Epsilon-Decay
arg_parser.add_argument('-e', '--epsilon', type=float, default=1)
arg_parser.add_argument('--e_min', type=float, default=0.1)
arg_parser.add_argument('--e_decay', type=float, default=0.0005)

## TODO find out whether 16,32, 64 is better for batch size
arg_parser.add_argument('--batch_size', type=int, default=16)
arg_parser.add_argument('--discount', type=float, default=0.98)
arg_parser.add_argument('--gradient_update_frequency', type=int, default= 1000)
arg_parser.add_argument('--step_max', type=int, default=250)
arg_parser.add_argument('--num_episodes', type=int, default=20)
arg_parser.add_argument('--training_mode', default='normal')
arg_parser.add_argument('--cuda', action='store_true')
arg_parser.add_argument('--learning_rate', type=float, default=0.0001)
arg_parser.add_argument('--render_training', action='store_true')
arg_parser.add_argument('--weight_path', type=str, default="")
arg_parser.add_argument('--save_weights', type=bool, default=False)
arg_parser.add_argument('--use_existing_weights', type=bool, default=False)
## TODO play with buffer_size and experience_replay_frequency
arg_parser.add_argument('--buffer_size', type=int, default=100000)
arg_parser.add_argument('--experience_replay_frequency', type=int, default=10)
arg_parser.add_argument('--use_target_net', default=True)
# Arguments for Prioritized Experience Replay
arg_parser.add_argument('--PER', default=False)
arg_parser.add_argument('--alpha', default=0.6)
arg_parser.add_argument('--beta', default=0.4)
arg_parser.add_argument('--beta_incremement', default=0.00001)
# TODO add further arguments when needed

## Short info: although I am using a Macbook with M1 chip which doesnt support cuda at all,
## I still included the option to use cuda in the code to also be able to run it with cuda support on another machine
parsed_args = arg_parser.parse_args()

if __name__ == '__main__':
    parsed_args.device = T.device('cuda' if parsed_args.cuda and T.cuda.is_available() else 'cpu')
    print("device: " + str(parsed_args))
    print(str(parsed_args.device.type))

    if parsed_args.training_mode == 'normal':
        mode = env.HockeyEnv_BasicOpponent.NORMAL
    elif parsed_args.training_mode == 'shooting':
        mode = env.HockeyEnv_BasicOpponent.TRAIN_SHOOTING
    elif parsed_args.training_mode == 'defense':
        mode = env.HockeyEnv_BasicOpponent.TRAIN_DEFENSE

    # Create Log Directory and pass it to the Logger
    config = vars(parsed_args)
    directory_name = time.strftime("%Y%m%d-%H%M%S") + '_' + str(config['num_episodes']) + '_' + str(
        config['training_mode'])
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', directory_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    logger = logger.Logger(log_directory=dir, config=config)

    # Save Hyperparameters
    logger.save_hyperparams_to_file(e=str(config['epsilon']),
                                    e_min=str(config['e_min']),
                                    e_decay=str(config['e_decay']),
                                    batch_size=str(config['batch_size']),
                                    discount=str(config['discount']),
                                    grd_upd_frq=str(config['gradient_update_frequency']),
                                    step_max=str(config['step_max']),
                                    num_episodes=str(config['num_episodes']),
                                    training_mode=str(config['training_mode']),
                                    learning_rate=str(config['learning_rate']),
                                    buffer_size=str(config['buffer_size']),
                                    exp_rep_frq=str(config['experience_replay_frequency']),
                                    use_target_net=str(config['use_target_net']),
                                    use_per=str(config['PER']),
                                    use_existing_weights=str(config['use_existing_weights']),
                                    weight_path=str(config['weight_path'])
                                    )

    # Set environment
    env = env.HockeyEnv(mode=mode)

    # Create an agent
    agent = DQNAgent(logger=logger,
                     config=config,
                     observation_space=env.observation_space.shape[0],
                     action_space=env.action_space
                     )

    trainer = Trainer(logger=logger, config=config, agent=agent, log_directory=dir)
    trainer.train(env=env, logger=logger)
