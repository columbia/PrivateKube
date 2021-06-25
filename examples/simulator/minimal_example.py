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

# There are two key concepts in the simulation program:
# 1. The simulation model: This implements how different components in the systems behave and interact with each other.
# 2. The configuration dictionary: a dictionary that specify the many aspects of simulation. details of this config dict is in minimal_example.ipynb
#
# The minimal example contains two steps:
#  1. Preparing the config dictionary
#  2. Calling `simulate(config, Top)`, where `config` is the config dict and `Top` is the simulation model.


import sys
import os
from dpsched.utils.configs import DpPolicyType
from dpsched.DP_simulator import Top
from desmod.simulation import simulate
from datetime import datetime

# calculated parameters for convenient configuration.
dp_arrival_itvl = 0.078125
rdp_arrival_itvl = 0.004264781
N_dp = 100
T_dp = rdp_arrival_itvl * N_dp
N_rdp = 14514
T_rdp = rdp_arrival_itvl * N_rdp

config = {
    'workload_test.enabled': False,  # if enabled, generate workload from trace file.
    'workload_test.workload_trace_file': '/home/tao2/projects/PrivacySchedSim/workloads.yaml',
    # config of task workloads/demand
    'task.timeout.enabled': True,  # whether a waiting task should give up after timeout period
    'task.timeout.interval': 51,  # timeout period in sim.timescale
    'task.arrival_interval': dp_arrival_itvl,  # tasks' average arrival interval

    'task.demand.num_blocks.mice': 1,  # the minimum amount of data blocks each task may demands on.
    'task.demand.num_blocks.elephant': 10,  # the maximum amount of data blocks each task may demands on.
    'task.demand.num_blocks.mice_percentage': 75.0,  # the fraction of tasks with small demand in # of blocks
    'task.demand.epsilon.mice': 1e-2,  # the minimum amount of epsilon DP each task may demands on.
    'task.demand.epsilon.elephant': 1e-1,  # the maximum amount of epsilon DP each task may demands on.
    'task.demand.epsilon.mice_percentage': 75.0,  # the fraction of tasks with small demand in epsilon DP budget

    'task.demand.completion_time.constant': 0,
    # each task takes that time interval to execute, if ==0, finish immediately.
    'task.demand.num_cpu.constant': 1,  # each task takes that amount of CPU to execute.
    'task.demand.size_memory.max': 412,  # upper bound of memory used for each task
    'task.demand.size_memory.min': 1,
    # lower bound of memory used for each task, the memory size follows a uniform distribution
    'task.demand.num_gpu.max': 3,  # upper bound of GPU used for each task
    'task.demand.num_gpu.min': 1,  # lower bound of GPU used for each task, the GPU usage follows a uniform distribution

    # config of available resources
    'resource_master.block.init_epsilon': 1.0,  # privacy budget for initial epsilon
    'resource_master.block.init_delta': 1.0e-6,
    # privacy budget for initial delta, only used for rdp simulation, should be small << 1
    'resource_master.block.arrival_interval': 10,  # The time interval it takes to generate a new privacy data block.
    'resource_master.block.is_static': False,  # whether new blocks are generated dynamically
    'resource_master.block.init_amount': 11,  # initial amount of data blocks before the simulation starts
    'resource_master.is_cpu_needed_only': True,  # if True, a task only grab CPU when running, NO memory GPU etc
    'resource_master.cpu_capacity': sys.maxsize,  # number of cores the system has for allocation.
    'resource_master.memory_capacity': 624,
    # memory capacity the system has for allocation. in GB, assume granularity is 1GB
    'resource_master.gpu_capacity': 8,  # number of GPU cards the system has for allocation.

    # config of scheduling policy
    'resource_master.dp_policy': DpPolicyType.DP_POLICY_DPF_N,  # the policy name of scheduler
    'resource_master.dp_policy.denominator': N_dp,  # parameter N for number of arrived tasks based scheduling policy
    'resource_master.block.lifetime': T_rdp,  # parameter T for lifetime based scheduling policy
    'resource_master.dp_policy.is_rdp': False,
    # whether the scheduler use Renyi DP composition or epsilon-delta DP composition
    'resource_master.dp_policy.dpf_family.grant_top_small': False,
    # If the scheduler use DPF-like policy, whether only grant first-k (leading smallest) tasks or do best effort grant until the last task.
    'resource_master.dp_policy.is_admission_control_enabled': False,
    # whether a task should be allowed to wait when any of its demanded block is retired.

    # general config of simulation
    'sim.duration': '300 s',  # simulated duration
    'sim.db.enable': True,  # whether record each tasks' key time points into the database.
    'sim.db.persist': True,  # whether persist the simulation result
    'sim.dot.colorscheme': 'blues5',
    'sim.dot.enable': False,
    'sim.runtime.timeout': 60,  # in min, the simulation program abort after this amount of interval.
    'sim.gtkw.file': 'sim.gtkw',  # gtkw file is metadata for display vcd file.
    'sim.gtkw.live': False,
    # live monitoring of resources duration simulation, through time series graph specified by gtkw file(metadata) .
    'sim.log.enable': True,  # enable log
    "sim.log.level": "DEBUG",
    'sim.progress.enable': True,  # display progress bar in terminal
    'sim.result.file': 'result.json',  # json file name for main result
    'sim.seed': 23338,  # random seed for simulation
    'sim.timescale': 's',  # time unit for simulation
    'sim.vcd.dump_file': 'sim_dp.vcd',
    'sim.vcd.enable': True,
    # a vcd file records how many resources change from different state (allocated, granted committed etc)
    'sim.vcd.persist': True,  # save vcd file,
    'sim.workspace': 'exp_results/workspace_%s' % datetime.now().strftime("%m-%d-%HH-%M-%S"),
    # simulation runs with workspace as its working directory, its results file are saved in workspace
    'sim.workspace.overwrite': True,  # overwrite existing workspace
    'sim.clock.adaptive_tick': True,
    # if True, the global clock ticks every 0.5*mean task arrival interval. otherwise tick every 0.1 sec
    'sim.main_file': os.path.abspath(__file__),
    'sim.numerical_delta': 1e-8,  # accuracy threshold for zero in numerical computation
    'sim.instant_timeout': 1e-8,  # timeout shortly after present
}

simulate(config, Top)
