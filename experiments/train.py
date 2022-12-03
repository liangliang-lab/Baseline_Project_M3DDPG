import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import sys
import os

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import m3ddpg.common.tf_util as U
from m3ddpg.trainer.m3ddpg import M3DDPGAgentTrainer
import tensorflow.contrib.layers as layers

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--bad-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--adv-eps", type=float, default=1e-3, help="adversarial training rate")
    parser.add_argument("--adv-eps-s", type=float, default=1e-5, help="small adversarial training rate")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-name", type=str, default="", help="name of which training state and model are loaded, leave blank to load seperately")
    parser.add_argument("--load-good", type=str, default="", help="which good policy to load")
    parser.add_argument("--load-bad", type=str, default="", help="which bad policy to load")
    ##### Noise for observation
    ## Gaussian
    parser.add_argument("--obs-gaus-std", type=float, help="mean of Gaussian noise for observation")
    parser.add_argument("--obs-gaus-mean", type=float, help="mean of Gaussian noise for observation")
    ## Uniform
    parser.add_argument("--obs-unif-high", type=float, help="Upper bound of the uniform interval of the noise for observation")
    parser.add_argument("--obs-unif-low", type=float, help="Lower bound of the uniform interval of the noise for observation")
    ## Laplace
    parser.add_argument("--obs-laplace-mean", type=float, help="mean of Laplace noise for observation")
    parser.add_argument("--obs-laplace-decay", type=float, help="decay of Laplace noise for observation")
    ## Beta
    parser.add_argument("--obs-beta-a", type=float, help="a of beta noise for observation")
    parser.add_argument("--obs-beta-b", type=float, help="b of beta noise for observation")
    ## Gamma
    parser.add_argument("--obs-gamma-shape", type=float, help="shape of gamma noise for observation")
    parser.add_argument("--obs-gamma-scale", type=float, help="scale of gamma noise for observation")
    ## Gumbel
    parser.add_argument("--obs-gumbel-mode", type=float,help="mode of gumbel noise for observation")
    parser.add_argument("--obs-gumbel-scale", type=float, help="scale of gumbel noise for observation")
    ## Wald
    parser.add_argument("--obs-wald-mean", type=float, help="mean of wald noise for observation")
    parser.add_argument("--obs-wald-scale", type=float, help="scale of wald noise for observation")
    ## logistic
    parser.add_argument("--obs-logistic-mean", type=float, help="mean of logistic noise for observation")
    parser.add_argument("--obs-logistic-scale", type=float, help="scale of logistic noise for observation")
    ##### Add different noise for actions
    ## Gaussian
    parser.add_argument("--act-gaus-std", type=float, help="std of Gaussian noise for action")
    parser.add_argument("--act-gaus-mean", type=float, help="mean of Gaussian noise level for action")
    ## Uniform
    parser.add_argument("--act-unif-high", type=float, help="Upper bound of the uniform interval of the noise for action")
    parser.add_argument("--act-unif-low", type=float, help="Lower bound of the uniform interval of the noise for action")
    ## Laplace
    parser.add_argument("--act-laplace-mean", type=float, help="mean of Laplace noise for action")
    parser.add_argument("--act-laplace-decay", type=float, help="decay of Laplace noise for action")
    ## Beta
    parser.add_argument("--act-beta-a", type=float, help="a of beta noise for action")
    parser.add_argument("--act-beta-b", type=float, help="b of beta noise for action")
    ## Gamma
    parser.add_argument("--act-gamma-shape", type=float, help="shape of gamma noise for action")
    parser.add_argument("--act-gamma-scale", type=float, help="scale of gamma noise for action")
    ## Gumbel
    parser.add_argument("--act-gumbel-mode", type=float,help="mode of gumbel noise for action")
    parser.add_argument("--act-gumbel-scale", type=float, help="scale of gumbel noise for action")
    ## Wald
    parser.add_argument("--act-wald-mean", type=float, help="mean of wald noise for action")
    parser.add_argument("--act-wald-scale", type=float, help="scale of wald noise for action")
    ## logistic
    parser.add_argument("--act-logistic-mean", type=float, help="mean of logistic noise for action")
    parser.add_argument("--act-logistic-scale", type=float, help="scale of logistic noise for action")
    
    
    # Evaluation
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = M3DDPGAgentTrainer
    for i in range(num_adversaries):
        print("{} bad agents".format(i))
        policy_name = arglist.bad_policy
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            policy_name == 'ddpg', policy_name, policy_name == 'mmmaddpg'))
    for i in range(num_adversaries, env.n):
        print("{} good agents".format(i))
        policy_name = arglist.good_policy
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            policy_name == 'ddpg', policy_name, policy_name == 'mmmaddpg'))
    return trainers


def train(arglist):
    if arglist.test:
        np.random.seed(71)
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and bad policy {} with {} adversaries'.format(arglist.good_policy, arglist.bad_policy, num_adversaries))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.test or arglist.display or arglist.restore or arglist.benchmark:
            if arglist.load_name == "":
                # load seperately
                bad_var_list = []
                for i in range(num_adversaries):
                    bad_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=trainers[i].scope)
                saver = tf.train.Saver(bad_var_list)
                U.load_state(arglist.load_bad, saver)

                good_var_list = []
                for i in range(num_adversaries, env.n):
                    good_var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=trainers[i].scope)
                saver = tf.train.Saver(good_var_list)
                U.load_state(arglist.load_good, saver)
            else:
                print('Loading previous state from {}'.format(arglist.load_name))
                U.load_state(arglist.load_name)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        def add_noise_action(arglist):
            if arglist.act_gaus_mean is not None and arglist.act_gaus_std is not None:
               return np.random.normal, arglist.act_gaus_mean, arglist.act_gaus_std
            if arglist.act_unif_low is not None and arglist.act_unif_high is not None:
               return np.random.uniform, arglist.act_unif_low, arglist.act_unif_high
            if arglist.act_laplace_mean is not None and arglist.act_laplace_decay is not None:
               return np.random.laplace, arglist.act_laplace_mean, arglist.act_laplace_decay
            if arglist.act_beta_a is not None and arglist.act_beta_b is not None:
               return np.random.beta, arglist.act_beta_a, arglist.act_beta_b
            if arglist.act_gamma_shape is not None and arglist.act_gamma_scale is not None:
               return np.random.gamma, arglist.act_gamma_shape, arglist.act_gamma_scale
            if arglist.act_gumbel_mode is not None and arglist.act_gumbel_scale is not None:
               return np.random.gumbel, arglist.act_gumbel_mode, arglist.act_gumbel_scale
            if arglist.act_wald_mean is not None and arglist.act_wald_scale is not None:
               return np.random.wald, arglist.act_wald_mean, arglist.act_wald_scale
            if arglist.act_logistic_mean is not None and arglist.act_logistic_scale is not None:
               return np.random.logistic, arglist.act_logistic_mean, arglist.act_logistic_scale
            
            return None, None, None

        def add_noise_observation(arglist):
            if arglist.obs_gaus_mean is not None and arglist.obs_gaus_std is not None:
               return np.random.normal, arglist.obs_gaus_mean, arglist.obs_gaus_std
            if arglist.obs_unif_low is not None and arglist.obs_unif_high is not None:
               return np.random.uniform, arglist.obs_unif_low, arglist.obs_unif_high
            if arglist.obs_laplace_mean is not None and arglist.obs_laplace_decay is not None:
               return np.random.laplace, arglist.obs_laplace_mean, arglist.obs_laplace_decay 
            if arglist.obs_beta_a is not None and arglist.obs_beta_b is not None:
               return np.random.beta, arglist.obs_beta_a, arglist.obs_beta_b
            if arglist.obs_gamma_shape is not None and arglist.obs_gamma_scale is not None:
               return np.random.gamma, arglist.obs_gamma_shape, arglist.obs_gamma_scale
            if arglist.obs_gumbel_mode is not None and arglist.obs_gumbel_scale is not None:
               return np.random.gumbel, arglist.obs_gumbel_mode, arglist.obs_gumbel_scale
            if arglist.obs_wald_mean is not None and arglist.obs_wald_scale is not None:
               return np.random.wald, arglist.obs_wald_mean, arglist.obs_wald_scale
            if arglist.obs_logistic_mean is not None and arglist.obs_logistic_scale is not None:
               return np.random.logistic, arglist.obs_logistic_mean, arglist.obs_logistic_scale

            return None, None, None

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]

            # add noise in actions
            act_noise_fun, act_noise_par1, act_noise_par2 = add_noise_action(arglist)
            if act_noise_fun is not None and act_noise_par1 is not None and act_noise_par2 is not None:
                for i, act in enumerate(action_n):
                    action_n[i] = act + act_noise_fun(act_noise_par1, act_noise_par2, act.shape)
            
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            obs_noise_fun, obs_noise_par1, obs_noise_par2 = add_noise_observation(arglist)
            if obs_noise_fun is not None and obs_noise_par1 is not None and obs_noise_par2 is not None:
                for i, obs_i in enumerate(new_obs_n):
                    new_obs_n[i] = obs_i + obs_noise_fun(obs_noise_par1, obs_noise_par2)

            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            if not arglist.test:
                loss = None
                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    loss = agent.update(trainers, train_step)

            # save model, display training output
            # if terminal and (len(episode_rewards) % arglist.save_rate == 0):
            #     U.save_state(arglist.save_dir, global_step = len(episode_rewards), saver=saver)
            #     # print statement depends on whether or not there are adversaries
            #     if num_adversaries == 0:
            #         print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
            #             train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
            #     else:
            #         print("{} vs {} steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(arglist.bad_policy, arglist.good_policy,
            #             train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
            #             [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
            #     t_start = time.time()
            #     # Keep track of final episode reward
            #     final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
            #     for rew in agent_rewards:
            #         final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            # if len(episode_rewards) > arglist.num_episodes:
            #     suffix = '_test.pkl' if arglist.test else '.pkl'
            #     rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards' + suffix
            #     agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards' + suffix

            #     if not os.path.exists(os.path.dirname(rew_file_name)):
            #         try:
            #             os.makedirs(os.path.dirname(rew_file_name))
            #         except OSError as exc:
            #             if exc.errno != errno.EEXIST:
            #                 raise

            #     with open(rew_file_name, 'wb') as fp:
            #         pickle.dump(final_ep_rewards, fp)
            #     with open(agrew_file_name, 'wb') as fp:
            #         pickle.dump(final_ep_ag_rewards, fp)
            #     print('...Finished total of {} episodes.'.format(len(episode_rewards)))
            #     break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)