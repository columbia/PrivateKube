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

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


def plot_granted_tasks(save_file_name, table, title, xaxis_col, yaxis_col):
    facetgrid = sns.relplot(data=table, x=xaxis_col, y=yaxis_col, row='is_rdp', col='N_or_T_based',
                            kind='line', hue='policy', style='policy', facet_kws=dict(sharex=False, sharey=False))

    plt.subplots_adjust(top=0.8)
    facetgrid.fig.suptitle(title)
    facetgrid.savefig(save_file_name)


def plot_delay_cdf(table, delay_lst_column, cdf_color_col, save_file_name, should_exclude_late_task,
                   should_modify_timeout_duration, task_timeout, plot_title):
    assert cdf_color_col in ('N_or_T_', 'epsilon_mice_fraction')
    table = table[['is_rdp', "policy", delay_lst_column, 'N_or_T_', 'epsilon_mice_fraction']]
    table_delay1 = pd.DataFrame({
        col: np.repeat(table[col].values, table[delay_lst_column].str.len())
        for col in table.columns.difference([delay_lst_column])
    })
    table_delay2 = pd.DataFrame(np.concatenate(table[delay_lst_column].values), columns=['start_time', 'commit_time'])
    table_delay = table_delay1.assign(commit_time=table_delay2['commit_time'],
                                      start_time=table_delay2['start_time'])  # [table1.columns.tolist()]
    table_delay.loc[table_delay['commit_time'].isna(), 'commit_time'] = np.Inf
    table_delay = table_delay.assign(
        dp_allocation_duration=lambda x: np.abs(x['commit_time']) - x['start_time'])  # [table1.columns.tolist()]

    if should_modify_timeout_duration == True:
        if isinstance(task_timeout, int):
            table_delay.loc[
                (table_delay['commit_time'] < 0) | table_delay['commit_time'].isin(
                    [np.Inf]), "dp_allocation_duration"] = task_timeout
        else:
            table_delay.loc[(table_delay['commit_time'] < 0) | table_delay['commit_time'].isin(
                [np.Inf]), "dp_allocation_duration"] = table_delay.loc[
                (table_delay['commit_time'] < 0) | table_delay['commit_time'].isna(), 'task_timeout']

    if should_exclude_late_task == True:
        table_delay = table_delay.loc[
            ~ ((table_delay['commit_time'] < 0) & (table_delay["dp_allocation_duration"] == 0))]
    if cdf_color_col == "N_or_T_":
        table_delay['log10_N_or_T_'] = table_delay['N_or_T_']
        table_delay.loc[table_delay['N_or_T_'] > 0, 'log10_N_or_T_'] = table_delay.loc[table_delay['N_or_T_'] > 0][
            'log10_N_or_T_'].apply(lambda x: round(math.log10(x), 3))
        g = sns.FacetGrid(data=table_delay, row='is_rdp', col="policy", hue='log10_N_or_T_', sharex=False,
                          palette='viridis')  # , palette=viridis )
    else:
        g = sns.FacetGrid(data=table_delay, row='is_rdp', col="policy", hue=cdf_color_col, sharex=False,
                          palette='viridis')  # , palette=viridis )
    g.map(sns.ecdfplot, "dp_allocation_duration")  # ,stat="count") #  )# ,palette=viridis  )
    g.add_legend()
    plt.subplots_adjust(top=0.8)
    g.fig.suptitle(plot_title)
    g.savefig(save_file_name)
