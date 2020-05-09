import yaml, os
import torch
from pathlib import Path


def save_args(args):
    with open(args.save_folder + '/args.yml', 'w') as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)


def evaluate_and_save(agent, env, args, best_reward, evaluate):
    eval_reward = evaluate(args, agent=agent, env=env)
    if eval_reward > best_reward:
        old_file = Path(args.save_folder + "/best_" + args.file_name + "_{:.3f}.p".format(best_reward))
        if old_file.exists():
            os.remove(old_file)
        best_reward = eval_reward
        agent.save_model(args.save_folder + "/best_" + args.file_name + "_{:.3f}".format(best_reward))
    return best_reward


def plot_grad(summary_writer, net, net_name, index):
    for i, param in enumerate(net.parameters()):
        name = "data/" + net_name + str(i) + "-absolute-grad"
        summary_writer.add_histogram(name, torch.abs(param.grad.detach()), index)


def plot_alpha(summary_writer, alpha, log_alpha, index):
    summary_writer.add_scalar("data/alpha", alpha, index)
    summary_writer.add_scalar("data/log-alpha", log_alpha, index)