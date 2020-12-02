import gym
from gnpy.core.utils import db2lin

from optical_rl_gym.gnpy_utils import propagation
from optical_rl_gym.envs.power_aware_rmsa_env import shortest_path_first_fit, \
    shortest_available_path_first_fit_fixed_power, \
    least_loaded_path_first_fit, SimpleMatrixObservation, PowerAwareRMSA
from optical_rl_gym.utils import evaluate_heuristic, random_policy

import pickle
import logging
import numpy as np

import matplotlib.pyplot as plt

load = 250
logging.getLogger('rmsacomplexenv').setLevel(logging.INFO)

seed = 20
episodes = 10
episode_length = 100

monitor_files = []
policies = []

# topology_name = 'gbn'
# topology_name = 'nobel-us'
# topology_name = 'germany50'
with open(f'../examples/topologies/germany50_eon_gnpy_5-paths.h5', 'rb') as f:
    topology = pickle.load(f)

env_args = dict(topology=topology, seed=10, allow_rejection=True, load=load, mean_service_holding_time=25,
                episode_length=episode_length, num_spectrum_resources=64)


class PowerAware_RMSA_Algorithm():

    def least_OPM_and_OBRM(self, env: PowerAwareRMSA) -> int:
        for idp, path in enumerate(env.k_shortest_paths[env.service.source, env.service.destination]):
            num_slots = env.get_number_slots(path)
            for initial_slot in range(0, env.topology.graph['num_spectrum_resources'] - num_slots):
                if env.is_path_free(path, initial_slot, num_slots):
                    min_osnr = env.k_shortest_paths[env.service.source, env.service.destination][idp].best_modulation[
                        "minimum_osnr"]
                    osnr = np.mean(propagation(db2lin(0) * 1e-3, 1, 1, env.service.source, env.service.destination,
                                               env.gnpy_network, env.eqpt_library))
                    launch_power = db2lin(min_osnr - osnr) * 1e-3
                    print(launch_power)
                    action = [idp, initial_slot, launch_power]
                    return action

        min_osnr = env.k_shortest_paths[env.service.source, env.service.destination][idp].best_modulation[
            "minimum_osnr"]
        osnr = np.mean(propagation(db2lin(0) * 1e-3, 1, 1, env.service.source, env.service.destination,
                                   env.gnpy_network, env.eqpt_library))
        launch_power = db2lin((min_osnr - osnr)) * 1e-3
        print(launch_power)
        action = [idp, initial_slot, launch_power]
        return action


print('STR'.ljust(5), 'REW'.rjust(7), 'STD'.rjust(7))

init_env = gym.make('PowerAwareRMSA-v0', **env_args)
env_rnd = SimpleMatrixObservation(init_env)
mean_reward_rnd, std_reward_rnd = evaluate_heuristic(env_rnd, PowerAware_RMSA_Algorithm().least_OPM_and_OBRM, n_eval_episodes=episodes)
print('Rnd:'.ljust(8), f'{mean_reward_rnd:.4f}  {std_reward_rnd:>7.4f}')
print('Bit rate blocking:', (
            init_env.episode_bit_rate_requested - init_env.episode_bit_rate_provisioned) /
      init_env.episode_bit_rate_requested)
print('Request blocking:',
      (init_env.episode_services_processed - init_env.episode_services_accepted) / init_env.episode_services_processed)
print('Total power:', 10 * np.log10(init_env.total_power))
print('Average power:', 10 * np.log10(init_env.total_power / init_env.services_accepted))

# print('Throughput:', init_env.topology.graph['throughput'])
