#     Copyright (c) 2021. Tao Luo <tao.luo@columbia.edu>
#
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import os
import sys
from datetime import datetime
from itertools import product

from desmod.simulation import simulate_factors

from dpsched import Top
from dpsched.utils.configs import DpPolicyType

if __name__ == '__main__':
    dp_arrival_itvl = 0.078125
    rdp_arrival_itvl = 0.004264781

    N_dp = 100
    T_dp = rdp_arrival_itvl * N_dp

    N_rdp = 15000
    T_rdp = rdp_arrival_itvl * N_rdp

    run_dp = True
    run_rdp = True
    workspace_dir = 'exp_results/workspace_%s' % datetime.now().strftime("%m-%d-%HH-%M-%S")
    print(f"experiment results are saved in  {workspace_dir}")
    config = {
        'sim.main_file': os.path.abspath(__file__),
        'sim.numerical_delta': 1e-8,
        'sim.instant_timeout': 0.01,
        'workload_test.enabled': False,
        'workload_test.workload_trace_file': None,
        'task.demand.num_blocks.mice': 1,
        'task.demand.num_blocks.elephant': 10,
        'task.demand.num_blocks.mu': 20,
        'task.demand.num_blocks.sigma': 10,
        'task.demand.epsilon.mean_tasks_per_block': 15,
        'task.demand.epsilon.mice': 1e-2,  # N < 100
        'task.demand.epsilon.elephant': 1e-1,
        'task.completion_time.constant': 0,  # finish immediately
        'task.demand.num_cpu.constant': 1,  # int, [min, max]

        'resource_master.block.init_epsilon': 1.0,  # normalized
        'resource_master.block.init_delta': 1.0e-6,  # only used for rdp should be small < 1
        'resource_master.block.arrival_interval': 10,
        'resource_master.dp_policy.dpf_family.grant_top_small': False,
        # false: best effort alloc; true: only continous leading small tasks in queue are granted

        'sim.duration': '300 s',
        'task.timeout.interval': 300,
        'task.timeout.enabled': True,
        'task.arrival_interval': rdp_arrival_itvl,
        'resource_master.dp_policy.is_admission_control_enabled': False,
        'resource_master.dp_policy.is_rdp': True,
        'resource_master.dp_policy': DpPolicyType.DP_POLICY_DPF_N,
        'resource_master.dp_policy.denominator': N_rdp,
        'resource_master.block.lifetime': T_rdp,  # policy level param
        'resource_master.block.is_static': False,
        'resource_master.block.init_amount': 11,  # for block elephant demand
        'task.demand.num_blocks.mice_percentage': 75.0,
        'task.demand.epsilon.mice_percentage': 75.0,
        'resource_master.is_cpu_needed_only': True,
        'resource_master.cpu_capacity': sys.maxsize,  # number of cores
        'resource_master.memory_capacity': 624,  # in GB, assume granularity is 1GB
        'resource_master.gpu_capacity': 8,  # in cards
        'resource_master.clock.tick_seconds': 25,
        'sim.clock.adaptive_tick': True,
        'sim.db.enable': True,
        'sim.db.persist': True,
        'sim.dot.colorscheme': 'blues5',
        'sim.dot.enable': False,

        'sim.runtime.timeout': 60,  # in min

        'sim.gtkw.file': 'sim.gtkw',
        'sim.gtkw.live': False,
        'sim.log.enable': True,
        "sim.log.level": "DEBUG",
        'sim.progress.enable': True,
        'sim.result.file': 'result.json',
        'sim.seed': 3345,  # 23338,
        'sim.timescale': 's',
        'sim.vcd.dump_file': 'sim_dp.vcd',
        'sim.vcd.enable': False,
        'sim.vcd.persist': False,
        'sim.workspace': workspace_dir,
        'sim.workspace.overwrite': True,
    }
    ## single block setup for reproduce result of submission
    config['resource_master.block.is_static'] = True
    config['resource_master.block.init_amount'] = 1
    config['task.arrival_interval'] = 1  # unit
    config['task.demand.num_blocks.mice_percentage'] = 100

    mice_fraction = [0, 25, 50, 75, 100]


    class Config(object):
        def __init__(self):
            pass


    config_list = []
    # pure DP
    dp_max_amount = 100
    dp_subconfig = Config()
    dp_subconfig.is_rdp = False
    N_scale_factor = [0.10, 0.50, 0.75, 1.00, 1.25, 1.75, 2.00, 2.25, 2.75, 3.25]
    num_arrivals_multiplier = (
            2.25 * 1.05
    )
    dp_subconfig.dp_policy = [
        DpPolicyType.DP_POLICY_FCFS,
        DpPolicyType.DP_POLICY_DPF_T,
        DpPolicyType.DP_POLICY_DPF_N,
        DpPolicyType.DP_POLICY_RR_T,
        DpPolicyType.DP_POLICY_RR_N,
    ]
    DP_N = dp_subconfig.denominator = [1] + [dp_max_amount * i for i in N_scale_factor]
    DP_T = dp_subconfig.block_lifetime = [
        N * config['task.arrival_interval'] for N in DP_N
    ]
    dp_subconfig.sim_duration = '%d s' % (
            config['task.arrival_interval'] * dp_max_amount * num_arrivals_multiplier
    )

    dp_timeout = (
            3 * 100 * config['task.arrival_interval']
    )  # at max, 100 tasks waiting in the queue
    if run_dp:
        for p in dp_subconfig.dp_policy:
            if p == DpPolicyType.DP_POLICY_FCFS:
                config_list.extend(
                    list(
                        product(
                            [dp_timeout],
                            [dp_subconfig.is_rdp],
                            [dp_subconfig.sim_duration],
                            [p],
                            [None],
                            [None],
                        )
                    )
                )
            elif p == DpPolicyType.DP_POLICY_DPF_T:
                config_list.extend(
                    list(
                        product(
                            [dp_timeout],
                            [dp_subconfig.is_rdp],
                            [dp_subconfig.sim_duration],
                            [p],
                            [None],
                            DP_T,
                        )
                    )
                )
            elif p == DpPolicyType.DP_POLICY_DPF_N:
                config_list.extend(
                    list(
                        product(
                            [dp_timeout],
                            [dp_subconfig.is_rdp],
                            [dp_subconfig.sim_duration],
                            [p],
                            DP_N,
                            [None],
                        )
                    )
                )
            elif p == DpPolicyType.DP_POLICY_RR_T:
                config_list.extend(
                    list(
                        product(
                            [dp_timeout],
                            [dp_subconfig.is_rdp],
                            [dp_subconfig.sim_duration],
                            [p],
                            [None],
                            DP_T,
                        )
                    )
                )
            elif p == DpPolicyType.DP_POLICY_RR_N:
                config_list.extend(
                    list(
                        product(
                            [dp_timeout],
                            [dp_subconfig.is_rdp],
                            [dp_subconfig.sim_duration],
                            [p],
                            DP_N,
                            [None],
                        )
                    )
                )
            else:
                raise Exception()

    # RDP
    is_rdp = True
    rdp_max_amount = 14514

    rdp_subconfig = Config()
    rdp_subconfig.is_rdp = True

    rdp_subconfig.dp_policy = [DpPolicyType.DP_POLICY_FCFS, DpPolicyType.DP_POLICY_DPF_T, DpPolicyType.DP_POLICY_DPF_N]
    RDP_N = rdp_subconfig.denominator = [1] + [
        int(rdp_max_amount * n) for n in N_scale_factor
    ]
    RDP_T = rdp_subconfig.block_lifetime = [
        N * config['task.arrival_interval'] for N in RDP_N
    ]
    rdp_subconfig.sim_duration = '%d s' % (
            config['task.arrival_interval'] * rdp_max_amount * num_arrivals_multiplier
    )

    rdp_timeout = (
            3 * 100 * config['task.arrival_interval']
    )  # at max, 100 tasks waiting in the queue
    if run_rdp:
        for p in rdp_subconfig.dp_policy:
            if p == DpPolicyType.DP_POLICY_FCFS:
                config_list.extend(
                    list(
                        product(
                            [rdp_timeout],
                            [rdp_subconfig.is_rdp],
                            [rdp_subconfig.sim_duration],
                            [p],
                            [None],
                            [None],
                        )
                    )
                )
            elif p == DpPolicyType.DP_POLICY_DPF_T:
                config_list.extend(
                    list(
                        product(
                            [rdp_timeout],
                            [rdp_subconfig.is_rdp],
                            [rdp_subconfig.sim_duration],
                            [p],
                            [None],
                            RDP_T,
                        )
                    )
                )
            elif p == DpPolicyType.DP_POLICY_DPF_N:
                config_list.extend(
                    list(
                        product(
                            [rdp_timeout],
                            [rdp_subconfig.is_rdp],
                            [rdp_subconfig.sim_duration],
                            [p],
                            RDP_N,
                            [None],
                        )
                    )
                )
            else:
                raise Exception()

    real_config_fields = [
        'task.timeout.interval',
        'resource_master.dp_policy.is_rdp',
        'sim.duration',
        'resource_master.dp_policy',
        'resource_master.dp_policy.denominator',
        'resource_master.block.lifetime',
    ]

    test_factors = [
        (real_config_fields, config_list),
        (['task.demand.epsilon.mice_percentage'], [[pct] for pct in mice_fraction]),
    ]
    load_filter = lambda x: True
    simulate_factors(config, test_factors, Top, config_filter=load_filter)
