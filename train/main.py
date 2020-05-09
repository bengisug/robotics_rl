import argparse
import importlib
import os
from datetime import datetime as dt
from train import config


def get_parser():
    time_sign = str(dt.now()).replace(' ', '').replace(':', '')
    parser = argparse.ArgumentParser(
        description="Baxter arm manipulation training with Deep Reinforcement Learning!")

    parser.add_argument("--lr-actor", type=float, default=0.0003,
                        help="Learning rate of actor (default: %(default)s)")
    parser.add_argument("--lr-critic", type=float, default=0.0003,
                        help="Learning rate of critic (default: %(default)s)")
    parser.add_argument("--gamma", type=float, default=0.985,
                        help="Discount rate (default: %(default)s)")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Target update rate (default: %(default)s)")
    parser.add_argument("--theta", type=float, default=0.15,
                        help=("Drift value in Ornstein Uhlenbeck random "
                              " process (default: %(default)s)"))
    parser.add_argument("--clip-grad", action="store_true",
                        help=("Clip gradients if True"
                              " (default: %(default)s)"))
    parser.add_argument("--start-sigma", type=float, default=0.2,
                        help=("Initial noise variance value "
                              "(default: %(default)s)"))
    parser.add_argument("--end-sigma", type=float, default=0.01,
                        help=("Terminal noise variance value "
                              "(default: %(default)s)"))
    parser.add_argument("--start-alpha", type=float, default=0.2,
                        help=("Initial noise variance value "
                              "(default: %(default)s)"))
    parser.add_argument("--end-alpha", type=float, default=0.01,
                        help=("Terminal noise variance value "
                              "(default: %(default)s)"))
    parser.add_argument("--episodes", type=int, default=10000,
                        help="Number of episodes (default: %(default)s)")
    parser.add_argument("--max-length", type=int, default=400,
                        help="Maximum time step of the environment (default: %(default)s)")
    parser.add_argument("--buffer-size", type=int, default=1000000,
                        help="Replay buffer size (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size (default: %(default)s)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Pytorch device (default: %(default)s)",
                        choices=["cuda", "cpu"])
    parser.add_argument("--file-name", type=str, default="agent",
                        help="File name (default: %(default)s)")
    parser.add_argument("--control-type", type=str, default="position",
                        help="Baxter control type (default: %(default)s)",
                        choices=["position", "torque", "velocity"])
    parser.add_argument("--render", action="store_true",
                        help=("Render with opengl if True"
                              " (default: %(default)s)"))
    parser.add_argument("--agent-type", type=str, default="SACAgent",
                        help="Agent type (default: %(default)s)",
                        choices=["DDPGAgent", "SACAgent"])
    parser.add_argument("--buffer-type", type=str, default="UniformBuffer",
                        help="Buffer type (default: %(default)s)",
                        choices=["UniformBuffer", "PrioritizedBuffer"])
    parser.add_argument("--reward-type", type=str, default="shaped",
                        help="Reward type (default: %(default)s)",
                        choices=["shaped", "sparse"])
    parser.add_argument("--environment-type", type=str, default="ReachEnv1",
                        help="Environment type (default: %(default)s)",
                        choices=["ReachEnv1", "ReachEnv2", "PushEnv1", "PushEnv2", "PushEnv3", "GraspEnv1",
                                 "GraspEnv2"])
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the model (default: False)")
    parser.add_argument("--eval-period", type=int, default=50,
                        help=("Evaluate the model every nth episode "
                              "(default: %(default)s)"))
    parser.add_argument("--save-folder", type=str,
                        default='saved_results/' + time_sign,
                        help="Folder name for experiment results and logs (default: %(default)s)")
    parser.add_argument("--plot", action="store_true",
                        help=("Plot episode rewards and losses if True"
                              " (default: %(default)s)"))
    parser.add_argument("--plot-grad", action="store_true",
                        help=("Plot gradients if True"
                              " (default: %(default)s)"))
    parser.add_argument("--writer-name", type=str, default='saved_results/' + time_sign + '/plots',
                        help="Summary writer name (default: %(default)s)")
    parser.add_argument("--hindsight", action="store_true",
                        help="Hindsight (default: False)")
    parser.add_argument("--hindsight-mode", type=str, default="future",
                        help="Hindsight Mode (default: %(default)s)",
                        choices=["future", "success"])
    parser.add_argument("--hidden-layer-size", type=int, default=2,
                        help=("Hidden layer size for networks "
                              "(default: %(default)s)"))
    parser.add_argument("--node-size", type=int, default=256,
                        help=("Hidden node size for networks "
                              "(default: %(default)s)"))
    parser.add_argument("--range-coef", type=float, default=0.4,
                        help=("Range coefficient for push target range for PushEnv1 (between -1:1) "
                              "(default: %(default)s)"))
    parser.add_argument("--angle-coef", type=float, default=0.25,
                        help=("Angle coefficient for push target angle for PushEnv1 (between -1:1) "
                              "(default: %(default)s)"))

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_parser()
    cfg = config.Config(args)
    train_module = importlib.import_module(cfg.train_file)
    if args.evaluate:
        train_module.evaluate(args)
    else:
        os.mkdir(args.save_folder)
        train_module.train(args)