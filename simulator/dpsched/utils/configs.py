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

from enum import Enum

ENABLE_OSDI21_ARTIFACT_ONLY = True
DISABLED_FLAG = "DISABLED_FEATURE"

class DpPolicyType(str,Enum):
    DP_POLICY_FCFS = "FCFS"
    DP_POLICY_RR_T = "RR_T" # unlock/release quota evenly until lifetime in round robin fashion
    DP_POLICY_RR_N = "RR_N"  # rrt variant with unlock/release happens on task arrival
    DP_POLICY_DPF_N = "DPF_N"
    DP_POLICY_DPF_T = "DPF_T"
    # for disable following feature for OSDI artifact release
    if not ENABLE_OSDI21_ARTIFACT_ONLY:
        DP_POLICY_DPF_NA = "DPF_N_A"
        DP_POLICY_RR_N2 = "RR_N2" # unlock dp quota to the top N smallest tasks upon block end of life
    else:
        DP_POLICY_DPF_NA = DISABLED_FLAG
        DP_POLICY_RR_N2 = DISABLED_FLAG


class DpHandlerMessageType(str, Enum):
    ALLOCATION_SUCCESS = "V"
    ALLOCATION_FAIL = "F"
    ALLOCATION_REQUEST = "allocation_request"
    NEW_TASK = "new_task_created"
    DP_HANDLER_INTERRUPT_MSG = "interrupted_by_dp_hanlder"


class ResourceHandlerMessageType(str, Enum):
    RESRC_HANDLER_INTERRUPT_MSG = 'interrupted_by_resource_hanlder'
    RESRC_RELEASE = "released_resource"
    RESRC_PERMITED_FAIL_TO_ALLOC = "RESRC_PERMITED_FAIL_TO_ALLOC"
    RESRC_TASK_ARRIVAL = "RESRC_SCHED_TASK_ARRIVAL"

# alpha subsamples for budget curve of Renyi DP
ALPHAS = [1.000001, 1.0001, 1.5, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 20480]
TIMEOUT_VAL = "timeout_triggered"
DELTA = 1.0e-9 # delta budget in epsilon-delta DP
