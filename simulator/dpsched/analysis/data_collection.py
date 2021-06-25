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
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from ..utils.configs import DpPolicyType

viridis = sns.color_palette("viridis")


def parse_json(d):
    assert d['sim.exception'] is None
    res = {}
    c = d['config']
    res['blocks_mice_fraction'] = c["task.demand.num_blocks.mice_percentage"]
    res['epsilon_mice_fraction'] = c["task.demand.epsilon.mice_percentage"]
    res['arrival_interval'] = c["task.arrival_interval"]
    res['policy'] = c["resource_master.dp_policy"]
    res['lifetime'] = c["resource_master.block.lifetime"]
    res['sim_index'] = c["meta.sim.index"]

    res['task_amount_N'] = c["resource_master.dp_policy.denominator"]
    res['task_timeout'] = c['task.timeout.interval']
    res['is_rdp'] = c['resource_master.dp_policy.is_rdp']

    res['granted_tasks_total'] = d["succeed_tasks_total"]

    res['num_granted_tasks_l_dp_l_blk'] = d["succeed_tasks_l_dp_l_blk"]  # large dp large block #
    res['num_granted_tasks_l_dp_s_blk'] = d["succeed_tasks_l_dp_s_blk"]  # large dp small block #
    res['num_granted_tasks_s_dp_l_blk'] = d["succeed_tasks_s_dp_l_blk"]
    res['num_granted_tasks_s_dp_s_blk'] = d["succeed_tasks_s_dp_s_blk"]

    res['granted_tasks_per_10sec'] = d["succeed_tasks_total"] * 10 / d["sim.time"]
    res['sim_duration'] = d["sim.time"]
    res['grant_delay_P50'] = d.get("dp_allocation_duration_Median")
    res['grant_delay_P99'] = d.get("dp_allocation_duration_P99")  # may be None
    res['grant_delay_avg'] = d["dp_allocation_duration_avg"]
    res['grant_delay_max'] = d["dp_allocation_duration_max"]
    res['grant_delay_min'] = d["dp_allocation_duration_min"]
    # res['dp_allocation_duration_list'] = d['dp_allocation_duration']

    return res


def workspace2dataframe(workspace_dir):
    data_files = list(os.walk(workspace_dir))
    table_data = []
    json_file = "result.json"
    sqlite_file = "sim.sqlite"
    for d in data_files:
        if not d[1]:  # subdirectory
            result_json = None
            for file in d[2]:
                if file.endswith('json'):
                    result_json = file
                elif file == 'err.yaml':
                    raise (Exception(d))

            with open(os.path.join(d[0], json_file)) as f:
                data = json.load(f)
                try:
                    parsed_d = parse_json(data)
                    with sqlite3.connect(os.path.join(d[0], sqlite_file)) as conn:
                        alloc_dur = conn.execute(
                            # "select (abs(dp_commit_timestamp) - start_timestamp) AS dp_allocation_duration  from tasks").fetchall()
                            "select start_timestamp,dp_commit_timestamp from tasks"
                        ).fetchall()
                        parsed_d['dp_allocation_duration_list'] = alloc_dur
                        err_alloc_dur = conn.execute(
                            "select start_timestamp,dp_commit_timestamp from tasks where abs(dp_commit_timestamp) < start_timestamp"
                        ).fetchall()
                        if not len(err_alloc_dur) == 0:
                            raise Exception(err_alloc_dur)

                except Exception as e:
                    print(e)
                    print(data['config']['resource_master.dp_policy'])
                    print(data['config']['meta.sim.index'])
                    print(data['config']["resource_master.block.lifetime"])
                    #                 pprint(data)
                    print('\n\n\n')

                    raise (e)
                table_data.append(parsed_d)
    table = pd.DataFrame(table_data)
    if len(table.columns) == 0:
        raise Exception("no data found")
    table['N_or_T_based'] = None
    is_n_based = lambda x: x in (
        DpPolicyType.DP_POLICY_DPF_N.value, DpPolicyType.DP_POLICY_RR_N.value, DpPolicyType.DP_POLICY_RR_N2.value)
    is_t_based = lambda x: x in (
        DpPolicyType.DP_POLICY_DPF_T.value, DpPolicyType.DP_POLICY_RR_T.value, DpPolicyType.DP_POLICY_DPF_NA.value)
    table.loc[table['policy'].apply(is_n_based), "N_or_T_based"] = 'N'
    table.loc[table['policy'].apply(is_t_based), "N_or_T_based"] = 'T'
    table["N_"] = table.apply(get_n, axis=1)
    table["N_or_T_"] = table["N_"]
    table["N_or_T_"].loc[table['N_or_T_based'] == 'T'] = table["lifetime"]

    return table


def get_n(row):
    if row['N_or_T_based'] == 'N':
        return row['task_amount_N']
    elif row['N_or_T_based'] == 'T':
        return round(row['lifetime'] / row['arrival_interval'])
    else:
        return -1  # for fcfs




