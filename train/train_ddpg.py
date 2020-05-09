import numpy as np
import torch
from tensorboardX import SummaryWriter
from train import config, writer
from robotics_rl.agent.network.ddpg_net import DDPGActor, DDPGCritic
from robotics_rl.agent.model.ddpg_agent import DDPGAgent
from robotics_rl.agent.replay_buffer.transition import DDPGTransition
from robotics_rl.agent.replay_buffer.uniform_buffer import UniformBuffer
from robotics_rl.agent.noise.oup_noise import OrnsteinUhlenbeckProcess


def train(args):
    writer.save_args(args)
    summary_writer = SummaryWriter(args.writer_name)
    cfg = config.Config(args)

    env = cfg.env
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    actor = DDPGActor(obs_size, action_size, hidden_layer_size=args.hidden_layer_size, node_size=args.node_size)
    critic = DDPGCritic(obs_size, action_size, hidden_layer_size=args.hidden_layer_size, node_size=args.node_size)
    random_process = OrnsteinUhlenbeckProcess(args.theta, args.start_sigma, action_size)
    agent = DDPGAgent(actor, critic, args.lr_actor, args.lr_critic, args.gamma, args.clip_grad,
                      cfg.buffer, cfg.transition_type, args.batch_size, args.tau, random_process,
                      args.start_sigma, args.end_sigma, 1 / (args.max_length * args.episodes),
                      args.device)

    best_reward = -np.inf

    for eps in range(args.episodes):

        state = env.reset()
        episode_reward = 0

        for t in range(args.max_length):
            torch_state = agent._totorch(state, torch.float32).view(1, -1)
            action = agent.act(torch_state)
            next_state, reward, done, _ = env.step(action.numpy())
            episode_reward += reward

            transition = (state, action, reward, next_state, [done])
            agent.push_transition(*transition)
            state = next_state

            if len(agent.transition_buffer) > args.batch_size:
                value_loss, policy_loss = agent.update()
                if args.plot:
                    summary_writer.add_scalar('data/value-loss', value_loss, t + eps * args.max_length)
                    summary_writer.add_scalar('data/policy-loss', policy_loss, t + eps * args.max_length)
                if args.plot_grad:
                    writer.plot_grad(summary_writer, agent.actor, "actor", t + eps * args.max_length)
                    writer.plot_grad(summary_writer, agent.critic, "critic", t + eps * args.max_length)

        if args.hindsight is True:
            agent.generate_hindsight(env.getHindsightReward)
            if len(agent.transition_buffer) > args.batch_size:
                for _ in range(4):
                    value_loss, policy_loss = agent.update()

        print("Progress: {:.2}%, episode: {}/{}, episode reward: {:.5}"
              .format(eps / args.episodes * 100, eps, args.episodes,
                      episode_reward))

        if (eps + 1) % args.eval_period == 0:
            agent.save_model(args.save_folder + "/" + args.file_name)
            best_reward = writer.evaluate_and_save(agent=agent, env=env, args=args, best_reward=best_reward,
                                                   evaluate=evaluate)

        if args.plot:
            summary_writer.add_scalar('data/training-reward', episode_reward, eps)

    agent.save_model(args.save_folder + "/" + args.file_name)
    summary_writer.export_scalars_to_json("./all_scalars.json")
    summary_writer.close()


def evaluate(args, agent=None, env=None):
    if agent is None or env is None:
        cfg = config.Config(args)

        env = cfg.env

        obs_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

        actor = DDPGActor(obs_size, action_size, hidden_layer_size=args.hidden_layer_size, node_size=args.node_size)
        critic = DDPGCritic(obs_size, action_size, hidden_layer_size=args.hidden_layer_size, node_size=args.node_size)
        transition_type = DDPGTransition
        transition_buffer = UniformBuffer(args.buffer_size, transition_type)
        random_process = OrnsteinUhlenbeckProcess(args.theta, args.start_sigma, action_size)
        agent = DDPGAgent(actor, critic, args.lr_actor, args.lr_critic, args.gamma, args.clip_grad,
                          transition_buffer, transition_type, args.batch_size, args.tau, random_process,
                          args.start_sigma, args.end_sigma, 1 / (args.max_length * args.episodes),
                          args.device)
        agent.load_model(args.save_folder + "/" +args.file_name)
        episodes = args.episodes
    else:
        episodes = 10

    sum_reward = 0

    for eps in range(episodes):

        state = env.reset()
        episode_reward = 0

        for t in range(args.max_length):
            torch_state = agent._totorch(state, torch.float32).view(1, -1)
            action = agent.act_greedy(torch_state)
            next_state, reward, done, _ = env.step(action.numpy())
            episode_reward += reward

            state = next_state
        sum_reward += episode_reward

    mean_reward = sum_reward / episodes

    print("Evaluation reward: {:.5}".format(mean_reward))

    return mean_reward

