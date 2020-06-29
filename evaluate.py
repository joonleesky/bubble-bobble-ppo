from common.env.atari_wrappers import wrap_deepmind
from common.model import MlpModel, NatureModel, ImpalaModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels

import pygame
import os, time, yaml, argparse
import gym, atari_py, retro
import torch

def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0, 0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='slow-49', help='experiment name')
    parser.add_argument('--env_name', type=str, default='BubbleBobble-Nes', help='environment ID')
    parser.add_argument('--param_name', type=str, default = 'BubbleBobble-Nes-Nature', help='hyper-parameter ID')
    parser.add_argument('--checkpoint', type=str, default='20000768', help='checkpoint number to load')
    parser.add_argument('--algo',       type=str, default = 'ppo', help='[a2c, ppo]')

    args = parser.parse_args()
    exp_name = args.exp_name
    env_name = args.env_name
    param_name = args.param_name
    checkpoint = args.checkpoint
    algo = args.algo
    device = torch.device("cpu")
    dir_path = './logs/' + env_name + '/' + algo + '/' + exp_name
    model_dir = os.path.join(dir_path, os.listdir(dir_path)[0])
    model_path = model_dir + '/model_' + checkpoint + '.pth'


    #########
    ## Env ##
    #########
    with open('./hyperparams/' + algo + '.yml', 'r') as f:
        hyperparameters = yaml.safe_load(f)[param_name]
    model_hyperparameters = hyperparameters.pop('model', {})
    agent_hyperparameters = hyperparameters

    env = retro.make(env_name, use_restricted_actions=retro.Actions.DISCRETE)
    env = wrap_deepmind(env)

    ###########
    ## Model ##
    ###########
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    action_space = env.action_space
    action_size = action_space.n
    architecture = hyperparameters.get('architecture', 'nature')

    if len(observation_shape) == 3:
        if architecture == 'nature':
            embedder = NatureModel(in_channels=observation_shape[0])
        elif architecture == 'impala':
            embedder = ImpalaModel(in_channels=observation_shape[0])
    elif len(observation_shape) == 1:
        embedder = MlpModel(input_dims=observation_shape[0],
                                **model_hyperparameters)

    policy = CategoricalPolicy(embedder, action_size)
    saved_model = torch.load(model_path)
    policy.load_state_dict(saved_model['state_dict'])
    policy.to(device)
    policy.eval()

    ###########
    ## AGENT ##
    ###########
    if algo == 'a2c':
        from agents.a2c import A2C as AGENT
    elif algo == 'ppo':
        from agents.ppo import PPO as AGENT

    agent = AGENT(env, policy, None, None, device, 0, **agent_hyperparameters)

    ##############
    ## EVALUATE ##
    ##############
    stages = ['01','11','21','31','41','51','61','71','81','91']
    for stage in stages:
        env.close()
        level = 'Level' + stage
        env = retro.make(env_name, use_restricted_actions=retro.Actions.DISCRETE, state=level)
        env = wrap_deepmind(env, episode_life=False)
        obs = env.reset()
        done = False
        lives = 3

        rendered = env.render(mode='rgb_array')
        zoom = 4
        fps = 50
        transpose = True
        video_size = [rendered.shape[1], rendered.shape[0]]
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)
        screen = pygame.display.set_mode(video_size)
        clock = pygame.time.Clock()

        while True:
            obs = torch.FloatTensor(obs).unsqueeze(0).to(device=device)
            act = policy.action(obs)
            next_obs, rew, done, info = env.step(act)
            obs = next_obs

            if obs is not None:
                rendered = env.render(mode='rgb_array')
                display_arr(screen, rendered, transpose=transpose, video_size=video_size)
            print(info)
            if info['enemies'] == 0:
                print(info)
                break
            if done == True:
                print(info)
                break

            pygame.display.flip()
            clock.tick(fps)
