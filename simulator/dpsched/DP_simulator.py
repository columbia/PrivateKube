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

import time
import heapq as hq

from datetime import timedelta
from functools import partial
from itertools import count, tee
import simpy
import timeit


import math
from dpsched.utils.rdp import (
    compute_rdp_epsilons_gaussian,
    gaussian_dp2sigma,
)
from operator import add, sub


from dpsched.utils.misc import max_min_fair_allocation, defuse
from dpsched.utils.configs import *
from dpsched.utils.store import (
    LazyAnyFilterQueue,
    DummyPutPool,
    DummyPool,
    DummyPutQueue,
    DummyPutLazyAnyFilterQueue,
    DummyFilterQueue,
)
from dpsched.utils.exceptions import *
import shutil
import os

from vcd.gtkw import GTKWSave

from typing import List

from pyDigitalWaveTools.vcd.parser import VcdParser
import yaml
from desmod.component import Component
from desmod.dot import generate_dot
import pprint as pp
import copy

NoneType = type(None)
IS_DEBUG = True

class Top(Component):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if ENABLE_OSDI21_ARTIFACT_ONLY:
            self._check_odsi_artifact_features_only()
        self._save_config()
        self.tasks = Tasks(self)
        self.resource_master = ResourceMaster(self)

        if self.env.config['sim.clock.adaptive_tick']:
            tick_seconds = (
                    self.env.config['task.arrival_interval'] * 0.5
            )  # median: avg arrival time x0.68
        else:
            tick_seconds = self.env.config['resource_master.clock.tick_seconds']
        self.global_clock = Clock(tick_seconds, self)
        self.add_process(self._timeout_stop)

    def _save_config(self):
        if self.env.config.get('sim.main_file'):
            shutil.copy(self.env.config['sim.main_file'],  os.path.basename(self.env.config['sim.main_file']) )
        if self.env.config['workload_test.enabled']:
            shutil.copy(self.env.config['workload_test.workload_trace_file'], os.path.join('./', os.path.basename(self.env.config['workload_test.workload_trace_file']) ))


    def _check_odsi_artifact_features_only(self):

        if self.env.config['resource_master.dp_policy'].value == DISABLED_FLAG:
            raise Exception("this feature is disabled for OSDI'21 artifact release, set ENABLE_OSDI21_ARTIFACT_ONLY in configs.py to enable it")
        else:
            return True

    def _timeout_stop(self):
        t0 = timeit.default_timer()
        while timeit.default_timer() - t0 < self.env.config['sim.runtime.timeout'] * 60:
            yield self.env.timeout(20)
        raise Exception(
            'Simulation timeout %d min ' % self.env.config['sim.runtime.timeout']
        )

    def connect_children(self):
        self.connect(self.tasks, 'resource_master')
        self.connect(self.tasks, 'global_clock')
        self.connect(self.resource_master, 'global_clock')

    @classmethod
    def pre_init(cls, env):

        with open(env.config['sim.gtkw.file'], 'w') as gtkw_file:
            gtkw = GTKWSave(gtkw_file)
            gtkw.dumpfile(env.config['sim.vcd.dump_file'], abspath=False)
            gtkw.treeopen('dp_sim')
            gtkw.signals_width(300)
            analog_kwargs = {
                'color': 'cycle',
                'extraflags': ['analog_step'],
            }
            with gtkw.group(f'task'):
                scope = 'tasks'
                gtkw.trace(f'{scope}.active_count', datafmt='dec', **analog_kwargs)
                gtkw.trace(f'{scope}.completion_count', datafmt='dec', **analog_kwargs)
                gtkw.trace(f'{scope}.fail_count', datafmt='dec', **analog_kwargs)

            with gtkw.group(f'resource'):
                scope = 'resource_master'
                gtkw.trace(f'{scope}.cpu_pool', datafmt='dec', **analog_kwargs)
                gtkw.trace(f'{scope}.gpu_pool', datafmt='dec', **analog_kwargs)
                gtkw.trace(f'{scope}.memory_pool', datafmt='dec', **analog_kwargs)
                gtkw.trace(f'{scope}.unused_dp', datafmt='real', **analog_kwargs)
                gtkw.trace(f'{scope}.committed_dp', datafmt='real', **analog_kwargs)

    def elab_hook(self):
        generate_dot(self)

    def get_result_hook(self,result):
        pass


class Clock(Component):
    base_name = 'clock'

    def __init__(self, seconds_per_tick, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.seconds_per_tick = seconds_per_tick
        self.ticking_proc_h = self.env.process(self.precise_ticking())

    def precise_ticking(self):
        tick_counter = count()
        while True:
            tick_event_time = next(tick_counter) * self.seconds_per_tick
            # high accurate tick, sched a timeout-like event
            tick_event = self.env.event()
            tick_event._value = None
            tick_event._ok = True
            self.env.schedule_at(event=tick_event, sim_time=tick_event_time)
            yield tick_event


    @property
    def next_tick(self):
        return self.ticking_proc_h.target


class ResourceMaster(Component):
    base_name = 'resource_master'


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_connections('global_clock')
        self._retired_blocks = set()
        self._avg_num_blocks = (
                                       self.env.config['task.demand.num_blocks.mice']
                                       + self.env.config['task.demand.num_blocks.elephant']
                               ) / 2
        self._avg_epsilon = (
                                    self.env.config['task.demand.epsilon.mice']
                                    + self.env.config['task.demand.epsilon.elephant']
                            ) / 2
        self.dp_policy = self.env.config["resource_master.dp_policy"]
        self.is_dp_policy_fcfs = self.dp_policy == DpPolicyType.DP_POLICY_FCFS
        self.is_rdp = self.env.config['resource_master.dp_policy.is_rdp']
        self.is_admission_control_enabled = self.env.config[
            'resource_master.dp_policy.is_admission_control_enabled'
        ]
        self.is_dp_policy_dpfn = self.dp_policy == DpPolicyType.DP_POLICY_DPF_N
        self.is_dp_policy_dpft = self.dp_policy == DpPolicyType.DP_POLICY_DPF_T
        self.is_dp_policy_dpfna = self.dp_policy == DpPolicyType.DP_POLICY_DPF_NA
        self.is_dp_dpf = (not self.is_rdp) and (
                self.is_dp_policy_dpfna or self.is_dp_policy_dpfn or self.is_dp_policy_dpft
        )
        self.is_rdp_dpf = self.is_rdp and (
                self.is_dp_policy_dpfna or self.is_dp_policy_dpfn or self.is_dp_policy_dpft
        )
        self.is_dp_policy_rr_t = self.dp_policy == DpPolicyType.DP_POLICY_RR_T
        self.is_dp_policy_rr_n2 = self.dp_policy == DpPolicyType.DP_POLICY_RR_N2
        self.is_dp_policy_rr_n = self.dp_policy == DpPolicyType.DP_POLICY_RR_N

        self.is_centralized_quota_sched = (
                self.is_dp_policy_dpfn or self.is_dp_policy_dpfna or self.is_dp_policy_dpft
        )
        self.is_accum_container_sched = (
                self.is_dp_policy_rr_t or self.is_dp_policy_rr_n2 or self.is_dp_policy_rr_n
        )
        self.is_N_based_retire = (
                self.is_dp_policy_rr_n or self.is_dp_policy_rr_n2 or self.is_dp_policy_dpfn
        )
        self.is_T_based_retire = (
                self.is_dp_policy_rr_t or self.is_dp_policy_dpft or self.is_dp_policy_dpfna
        )
        # regardless of rdp or dp
        self.does_task_handler_unlock_quota = (
                self.is_dp_policy_rr_n2 or self.is_dp_policy_dpfn or self.is_dp_policy_dpfna
        )

        if not self.is_rdp:
            if self.is_dp_policy_dpfn:
                pass
            elif self.is_dp_policy_dpft:
                pass
            elif self.is_dp_policy_dpfna:
                pass
            elif self.is_dp_policy_rr_n2:
                NotImplementedError()
            elif self.is_dp_policy_rr_n:
                pass
            elif self.is_dp_policy_rr_t:
                pass
            elif self.is_dp_policy_fcfs:
                pass
            else:
                raise NotImplementedError()
        else:
            assert self.is_rdp
            if self.is_dp_policy_dpfn:
                pass
            elif self.is_dp_policy_dpft:
                pass  # raise NotImplementedError()
            elif self.is_dp_policy_dpfna:
                raise NotImplementedError()
            elif self.is_dp_policy_rr_n2:
                raise NotImplementedError()
            elif self.is_dp_policy_rr_n:
                raise NotImplementedError()
            elif self.is_dp_policy_rr_t:
                raise NotImplementedError()
            elif self.is_dp_policy_fcfs:
                pass
            else:
                raise NotImplementedError()

        self.unused_dp = DummyPool(self.env)

        self.auto_probe('unused_dp', vcd={'var_type': 'real'})
        self.init_blocks_ready = self.env.event()
        self.committed_dp = DummyPool(self.env)

        self.auto_probe('committed_dp', vcd={'var_type': 'real'})

        self.block_dp_storage = DummyPutQueue(self.env, capacity=float("inf"))

        self.is_cpu_needed_only = self.env.config['resource_master.is_cpu_needed_only']
        self.cpu_pool = DummyPool(
            self.env,
            capacity=self.env.config["resource_master.cpu_capacity"],
            init=self.env.config["resource_master.cpu_capacity"],
            hard_cap=True,
        )
        self.auto_probe('cpu_pool', vcd={})

        self.memory_pool = DummyPool(
            self.env,
            capacity=self.env.config["resource_master.memory_capacity"],
            init=self.env.config["resource_master.memory_capacity"],
            hard_cap=True,
        )
        self.auto_probe('memory_pool', vcd={})

        self.gpu_pool = DummyPool(
            self.env,
            capacity=self.env.config["resource_master.gpu_capacity"],
            init=self.env.config["resource_master.gpu_capacity"],
            hard_cap=True,
        )
        self.auto_probe('gpu_pool', vcd={})

        self.mail_box = DummyPutQueue(self.env)
        # make sure get event happens at the last of event queue at current epoch.
        self.resource_sched_mail_box = DummyPutLazyAnyFilterQueue(self.env)
        # two types of item in mail box:
        # 1. a list of block ids whose quota get incremented,
        # 2. new arrival task id
        self.dp_sched_mail_box = DummyPutLazyAnyFilterQueue(self.env)

        self.task_state = dict()  # {task_id:{...},...,}
        self.add_processes(self.generate_datablocks_loop)
        self.add_processes(self.allocator_frontend_loop)
        self.debug("dp allocation policy %s" % self.dp_policy)
        # waiting for dp permission
        self.dp_waiting_tasks = DummyFilterQueue(
            self.env, capacity=float("inf")
        )  # {tid: DRS },  put task id and state to cal order

        # waiting for resource permission
        self.resource_waiting_tasks = DummyFilterQueue(self.env, capacity=float("inf"))

        self.auto_probe('resource_waiting_tasks', vcd={})
        self.auto_probe('dp_waiting_tasks', vcd={})

        if self.is_N_based_retire:
            self.denom = self.env.config['resource_master.dp_policy.denominator']
        else:
            self.denom = None
        # for quota based policy
        if self.is_centralized_quota_sched:
            self.add_processes(self.scheduling_dp_loop)

        self.add_processes(self.scheduling_resources_loop)

    def scheduling_resources_loop(self):
        def _permit_resource(request_tid, idle_resources):
            # non blocking
            # warning, resource allocation may fail/abort after permitted.
            # e.g. when resource handler is interrupted

            self.resource_waiting_tasks.get(filter=lambda x: x == request_tid)
            idle_resources['cpu_level'] -= self.task_state[request_tid][
                'resource_request'
            ]['cpu']
            if not self.is_cpu_needed_only:
                idle_resources['gpu_level'] -= self.task_state[request_tid][
                    'resource_request'
                ]['gpu']
                idle_resources['memory_level'] -= self.task_state[request_tid][
                    'resource_request'
                ]['memory']
            self.task_state[request_tid]['resource_permitted_event'].succeed()

        # fixme coverage
        def _reject_resource(request_tid):

            self.resource_waiting_tasks.get(filter=lambda x: x == request_tid)
            self.task_state[request_tid]['resource_permitted_event'].fail(
                RejectResourcePermissionError('xxxx')
            )

        while True:

            yield self.resource_sched_mail_box.when_any()
            # ensure the scheduler is really lazy to process getter
            assert (
                    self.env.peek() != self.env.now
                    or self.env._queue[0][1] == LazyAnyFilterQueue.LAZY
            )
            # ignore fake door bell, listen again
            if len(self.resource_sched_mail_box.items) == 0:
                continue

            mail_box = self.resource_sched_mail_box
            # HACK avoid calling slow get()
            msgs, mail_box.items = mail_box.items, []

            resrc_release_msgs = []
            new_arrival_msgs = []
            fail_alloc_msgs = []

            for msg in msgs:
                if msg['msg_type'] == ResourceHandlerMessageType.RESRC_TASK_ARRIVAL:
                    new_arrival_msgs.append(msg)
                elif msg['msg_type'] == ResourceHandlerMessageType.RESRC_RELEASE:
                    resrc_release_msgs.append(msg)
                # fixme coverage
                elif msg['msg_type'] == ResourceHandlerMessageType.RESRC_PERMITED_FAIL_TO_ALLOC:
                    fail_alloc_msgs.append(msg)
                else:
                    raise Exception('cannot identify message type')

            new_arrival_tid = [m['task_id'] for m in new_arrival_msgs]
            # should be a subset

            assert set(new_arrival_tid) <= set(self.resource_waiting_tasks.items)

            task_sched_order = None
            # optimization for case with only new arrival task(s), fcfs
            if len(new_arrival_msgs) == len(msgs):
                task_sched_order = new_arrival_tid
            # otherwise, iterate over all sleeping tasks to sched.
            else:
                task_sched_order = copy.deepcopy(self.resource_waiting_tasks.items)

            this_epoch_idle_resources = {
                "cpu_level": self.cpu_pool.level,
                "gpu_level": self.gpu_pool.level,
                "memory_level": self.memory_pool.level,
            }
            # save, sched later
            fcfs_sleeping_dp_waiting_tasks = []

            for sleeping_tid in task_sched_order:
                if not self.task_state[sleeping_tid]['dp_committed_event'].triggered:
                    # will schedule dp_waiting task later
                    fcfs_sleeping_dp_waiting_tasks.append(sleeping_tid)

                # sched dp granted tasks
                # first round: sched dp-granted tasks in FCFS order.
                elif self.task_state[sleeping_tid]['dp_committed_event'].ok:
                    if self._is_idle_resource_enough(
                            sleeping_tid, this_epoch_idle_resources
                    ):
                        _permit_resource(sleeping_tid, this_epoch_idle_resources)
                # fixme coverage
                else:
                    assert not self.task_state[sleeping_tid]['dp_committed_event'].ok
                    if not self.task_state[sleeping_tid]['is_admission_control_ok']:
                        _reject_resource(sleeping_tid)
                    else:
                        raise Exception(
                            "impossible to see dp rejected task in resource_waiting_tasks. This should already happen: failed dp commit -> "
                            "interrupt resoruce handler -> dequeue resource_waiting_tasks"
                        )

            # sched dp waiting tasks in FCFS order
            if (
                    self.is_dp_policy_fcfs
                    or self.is_dp_policy_rr_t
                    or self.is_dp_policy_rr_n2
                    or self.is_dp_policy_rr_n
            ):
                # regardless of rdp or dp
                sleeping_dp_waiting_sched_order = fcfs_sleeping_dp_waiting_tasks
            else:
                assert (
                        self.is_dp_policy_dpfn
                        or self.is_dp_policy_dpft
                        or self.is_dp_policy_dpfna
                )
                # regardless of rdp or dp
                # smallest dominant_resource_share task first
                sleeping_dp_waiting_sched_order = sorted(
                    fcfs_sleeping_dp_waiting_tasks,
                    reverse=False,
                    key=lambda t_id: self.task_state[t_id]['dominant_resource_share'],
                )

            # second round: sched dp ungranted
            for sleeping_tid in sleeping_dp_waiting_sched_order:
                if self._is_idle_resource_enough(
                        sleeping_tid, this_epoch_idle_resources
                ):
                    _permit_resource(sleeping_tid, this_epoch_idle_resources)


    def _is_mice_task_dp_demand(self, epsilon, num_blocks):
        return epsilon < self._avg_epsilon, num_blocks < self._avg_num_blocks

    def _is_idle_resource_enough(self, tid, idle_resources):
        if (
                idle_resources['cpu_level']
                < self.task_state[tid]['resource_request']['cpu']
        ):
            return False
        if not self.is_cpu_needed_only:
            if (
                    idle_resources['gpu_level']
                    < self.task_state[tid]['resource_request']['gpu']
            ):
                return False
            if (
                    idle_resources['memory_level']
                    < self.task_state[tid]['resource_request']['memory']
            ):
                return False

        return True


    def _cal_rdp_dominant_consumption(
            self, g_budget_curve, consumpiton_curve, demand_curve, g_budget_max
    ):
        current_rdp_max = max(map(sub, g_budget_curve, consumpiton_curve))
        post_alloc_consumption = map(add, consumpiton_curve, demand_curve)
        post_alloc_rdp_max = max(map(sub, g_budget_curve, post_alloc_consumption))
        return (current_rdp_max - post_alloc_rdp_max) / g_budget_max

    def _commit_rdp_allocation(self, block_idx: List[int], e_rdp: List[float]):
        assert len(block_idx) > 0

        for b in block_idx:
            # todo perf use iterator + map?
            temp_balance = []
            temp_quota_balance = []
            for j, e in enumerate(e_rdp):
                self.block_dp_storage.items[b]["rdp_consumption"][j] += e
                temp_balance.append(
                    self.block_dp_storage.items[b]["rdp_budget_curve"][j]
                    - self.block_dp_storage.items[b]["rdp_consumption"][j]
                )
                temp_quota_balance.append(
                    self.block_dp_storage.items[b]["rdp_quota_curve"][j]
                    - self.block_dp_storage.items[b]["rdp_consumption"][j]
                )
            assert max(temp_balance) >= (0 - self.env.config['sim.numerical_delta'])
            self.block_dp_storage.items[b]["rdp_quota_balance"] = temp_quota_balance

    def scheduling_dp_loop(self):
        # sourcery skip: hoist-statement-from-if, merge-duplicate-blocks, remove-redundant-if, simplify-len-comparison, split-or-ifs, swap-if-else-branches
        assert self.is_centralized_quota_sched
        # calculate DR share, match, allocate,
        # update DRS if new quota has over lap with tasks

        while True:
            doorbell = self.dp_sched_mail_box.when_any()
            yield doorbell
            # rejected or permitted tasks
            dp_processed_task_idx = []
            # ensure the scheduler is really lazy to process getter, wait for all quota incremented
            assert (
                    self.env.peek() != self.env.now
                    or self.env._queue[0][1] == LazyAnyFilterQueue.LAZY
            )
            # ignore fake door bell, listen again
            if len(self.dp_sched_mail_box.items) == 0:
                continue
            # HACK, avoid calling slow get()
            msgs, self.dp_sched_mail_box.items = self.dp_sched_mail_box.items, []

            new_arrival_tid = []
            incremented_quota_idx = set()
            msgs_amount = len(msgs)
            for m in msgs:
                if isinstance(m, int):
                    tid = m
                    # assert m in self.dp_waiting_tasks.items
                    idx = self.dp_waiting_tasks.items.index(tid, -msgs_amount - 10)
                    new_arrival_tid.append((idx, tid))
                else:
                    assert isinstance(m, list)
                    incremented_quota_idx.update(m)

            this_epoch_unused_quota = [
                block['dp_quota'].level for block in self.block_dp_storage.items
            ]
            # new task arrived
            for _, new_task_id in new_arrival_tid:
                assert self.task_state[new_task_id]['dominant_resource_share'] is None

            has_quota_increment =  len(incremented_quota_idx) > 0
            # update DRS of tasks if its demands has any incremented quota, or new comming tasks.
            quota_incre_upper_bound = (
                max(incremented_quota_idx) if has_quota_increment else -1
            )
            quota_incre_lower_bound = (
                min(incremented_quota_idx) if has_quota_increment else -1
            )
            # cal DRS
            if not self.is_rdp:
                self._cal_drs_dp_L_Inf(new_arrival_tid)

            else:
                assert self.is_rdp
                self._cal_drs_rdp_a_all2()

            permit_dp_task_order = None
            # optimization for no new quota case
            if (not has_quota_increment) and len(new_arrival_tid) != 0:
                new_arrival_drs = (
                    self.task_state[t[1]]['dominant_resource_share']
                    for t in new_arrival_tid
                )
                permit_dp_task_order = list(zip(new_arrival_drs, new_arrival_tid))
                hq.heapify(permit_dp_task_order)
            else:
                assert has_quota_increment
                waiting_task_drs = (
                    self.task_state[t]['dominant_resource_share']
                    for t in self.dp_waiting_tasks.items
                )
                permit_dp_task_order = list(
                    zip(waiting_task_drs, enumerate(self.dp_waiting_tasks.items))
                )
                hq.heapify(permit_dp_task_order)

            # iterate over tasks ordered by DRS, match quota, allocate.
            permitted_task_ids = set()
            dp_rejected_task_ids = set()
            permitted_blk_ids = set()
            should_grant_top_small = self.env.config[
                'resource_master.dp_policy.dpf_family.grant_top_small'
            ]
            are_leading_tasks_ok = True

            if not self.is_rdp:

                self._dpf_best_effort_dp_sched(
                    are_leading_tasks_ok,
                    dp_processed_task_idx,
                    permit_dp_task_order,
                    permitted_blk_ids,
                    permitted_task_ids,
                    should_grant_top_small,
                    this_epoch_unused_quota,
                )
                # reject tasks after allocation
                # only reject task on retired blocks
                if has_quota_increment:  # either dpft or dpfn
                    self._dpf_check_remaining_dp_n_reject(
                        dp_processed_task_idx,
                        dp_rejected_task_ids,
                        permitted_task_ids,
                        this_epoch_unused_quota,
                    )
            else:  # is_rdp
                self.best_effort_rdp_sched_n_commit_reject(
                    dp_processed_task_idx,
                    dp_rejected_task_ids,
                    permit_dp_task_order,
                    permitted_blk_ids,
                    permitted_task_ids,
                )

            # dequeue all permitted and rejected waiting tasks
            # HACK avoid calling dp_waiting_tasks.get()
            dp_processed_task_idx.sort(reverse=True)
            for i in dp_processed_task_idx:
                self.debug(
                    self.dp_waiting_tasks.items[i], "task get dequeued from wait queue"
                )
                del self.dp_waiting_tasks.items[i]


    def _dpf_check_remaining_dp_n_reject(
            self,
            dp_processed_task_idx,
            dp_rejected_task_ids,
            permitted_task_ids,
            this_epoch_unused_quota,
    ):
        # reject tasks after allocation
        # only reject task on retired blocks
        assert not self.is_rdp
        for idx, t_id in enumerate(self.dp_waiting_tasks.items):
            should_reject = None
            if t_id not in permitted_task_ids:
                this_task = self.task_state[t_id]
                this_request = this_task["resource_request"]
                task_demand_block_idx = this_request['block_idx']
                task_demand_epsilon = this_request['epsilon']

                # HACK, only check old and new items for rejection performance
                old_demand_b_idx = task_demand_block_idx[0]
                old_item = self.block_dp_storage.items[old_demand_b_idx]
                new_demand_b_idx = task_demand_block_idx[-1]
                # check oldest item this will check and reject for dpft
                if old_item["retire_event"].triggered:
                    assert old_item["retire_event"].ok
                    b = old_demand_b_idx
                    if this_epoch_unused_quota[b] < task_demand_epsilon:
                        should_reject = True
                # check latest item
                elif (
                        not self.is_dp_policy_dpft and new_demand_b_idx != old_demand_b_idx
                ):
                    new_item = self.block_dp_storage.items[new_demand_b_idx]
                    if new_item["retire_event"] and new_item["retire_event"].triggered:

                        b = new_demand_b_idx
                        if this_epoch_unused_quota[b] < task_demand_epsilon:
                            should_reject = True

                if should_reject:
                    this_task["dp_permitted_event"].fail(DpBlockRetiredError())
                    dp_rejected_task_ids.add(t_id)
                    dp_processed_task_idx.append(idx)

    def best_effort_rdp_sched_n_commit_reject(
            self,
            dp_processed_task_idx,
            dp_rejected_task_ids,
            permit_dp_task_order,
            permitted_blk_ids,
            permitted_task_ids,
    ):
        for drs, t in permit_dp_task_order:
            t_idx, t_id = t
            this_task = self.task_state[t_id]
            this_request = this_task["resource_request"]

            task_demand_block_idx = this_request['block_idx']

            task_demand_e_rdp = this_request['e_rdp']

            violated_blk, is_quota_insufficient_all = self.is_all_block_quota_sufficient(task_demand_block_idx, task_demand_e_rdp)

            # task is permitted
            if not is_quota_insufficient_all :#
                drs = this_task['dominant_resource_share']
                self.debug(t_id, "DP permitted, Dominant resource share: %.3f" % drs)
                this_task["dp_permitted_event"].succeed()
                permitted_task_ids.add(t_id)
                permitted_blk_ids.update(task_demand_block_idx)

                # need to update consumption for following rejection
                self._commit_rdp_allocation(task_demand_block_idx, task_demand_e_rdp)
                this_task["dp_committed_event"].succeed()
                this_task['is_dp_granted'] = True
                dp_processed_task_idx.append(t_idx)
            else: # is_quota_insufficient_all
                if self.block_dp_storage.items[violated_blk]["retire_event"].triggered:
                    assert self.block_dp_storage.items[violated_blk]["retire_event"].ok
                    dp_rejected_task_ids.add(t_id)

                    this_task["dp_permitted_event"].fail(
                        DpBlockRetiredError(
                            "block %d retired, insufficient unlocked rdp left" % violated_blk
                        )
                    )
                    this_task['is_dp_granted'] = False
                    dp_processed_task_idx.append(t_idx)

        return

    def is_all_block_quota_sufficient(self, task_demand_block_idx, task_demand_e_rdp):

        for b in task_demand_block_idx:
            for j, e_d in enumerate(task_demand_e_rdp):
                if (
                        e_d
                        <= self.block_dp_storage.items[b]['rdp_quota_balance'][j]
                ):
                    break
            else:
                return b, True

        else:

            return None, False

    def _dpf_best_effort_dp_sched(
            self,
            are_leading_tasks_ok,
            dp_processed_task_idx,
            permit_dp_task_order,
            permitted_blk_ids,
            permitted_task_ids,
            should_grant_top_small,
            this_epoch_unused_quota,
    ):
        for drs, t in permit_dp_task_order:
            t_idx, t_id = t
            if should_grant_top_small and (not are_leading_tasks_ok):
                break
            this_task = self.task_state[t_id]
            this_request = this_task["resource_request"]
            task_demand_block_idx = this_request['block_idx']

            task_demand_epsilon = this_request['epsilon']

            for b_idx in task_demand_block_idx:
                if (
                        this_epoch_unused_quota[b_idx]
                        + self.env.config['sim.numerical_delta']
                        < task_demand_epsilon
                ):
                    are_leading_tasks_ok = False
                    break
            # task is permitted
            else:
                drs = this_task['dominant_resource_share']
                self.debug(t_id, "DP permitted, Dominant resource share: %.3f" % drs)
                for i in task_demand_block_idx:
                    this_epoch_unused_quota[i] -= task_demand_epsilon
                this_task["dp_permitted_event"].succeed()
                permitted_task_ids.add(t_id)
                permitted_blk_ids.update(task_demand_block_idx)

                dp_processed_task_idx.append(t_idx)
        return

    def _cal_drs_rdp_a_all2(self):
        for t_id in reversed(self.dp_waiting_tasks.items):
            this_task = self.task_state[t_id]
            # ending condition, drs already calculated
            if this_task['dominant_resource_share'] is not None:
                break

            this_request = this_task['resource_request']
            # block wise
            temp_max = -1
            for b in this_request['block_idx']:
                for j, e in enumerate(this_request['e_rdp']):
                    # iterate over all alpha demand
                    if self.block_dp_storage.items[b]["rdp_budget_curve"][j] > 0:
                        normalized_e = (e / self.block_dp_storage.items[b]["rdp_budget_curve"][j]
                                        )
                        temp_max = max(temp_max, normalized_e)
            assert temp_max != -1
            this_task['dominant_resource_share'] = temp_max






    def _cal_drs_dp_L_Inf(self, new_arrival_tid):
        for _, new_task_id in new_arrival_tid:
            this_task = self.task_state[new_task_id]
            this_task['dominant_resource_share'] = this_task["resource_request"][
                'epsilon'
            ]


    def allocator_frontend_loop(self):
        while True:
            # loop only blocks here
            yield self.mail_box.when_any()

            for i in range(self.mail_box.size):
                get_evt = self.mail_box.get()
                msg = get_evt.value

                if msg["message_type"] == DpHandlerMessageType.NEW_TASK:
                    assert msg["task_id"] not in self.task_state
                    self.task_state[msg["task_id"]] = dict()
                    self.task_state[msg["task_id"]]["task_proc"] = msg["task_process"]

                if msg["message_type"] == DpHandlerMessageType.ALLOCATION_REQUEST:
                    assert msg["task_id"] in self.task_state
                    self.task_state[msg["task_id"]] = dict()
                    self.task_state[msg["task_id"]]["resource_request"] = msg
                    self.task_state[msg["task_id"]][
                        "resource_allocate_timestamp"
                    ] = None
                    self.task_state[msg["task_id"]]["dp_commit_timestamp"] = None
                    self.task_state[msg["task_id"]]["task_completion_timestamp"] = None
                    self.task_state[msg["task_id"]]["task_publish_timestamp"] = None

                    self.task_state[msg["task_id"]]["is_dp_granted"] = None
                    self.task_state[msg["task_id"]]["is_admission_control_ok"] = None
                    self.task_state[msg["task_id"]][
                        "resource_allocated_event"
                    ] = msg.pop("resource_allocated_event")
                    self.task_state[msg["task_id"]]["dp_committed_event"] = msg.pop(
                        "dp_committed_event"
                    )

                    # following two events are controlled by scheduling policy
                    self.task_state[msg["task_id"]][
                        "dp_permitted_event"
                    ] = self.env.event()
                    self.task_state[msg["task_id"]][
                        "resource_permitted_event"
                    ] = self.env.event()
                    self.task_state[msg["task_id"]][
                        "resource_released_event"
                    ] = self.env.event()

                    self.task_state[msg["task_id"]]["dominant_resource_share"] = None

                    self.task_state[msg["task_id"]]["execution_proc"] = msg.pop(
                        "execution_proc"
                    )
                    self.task_state[msg["task_id"]]["waiting_for_dp_proc"] = msg.pop(
                        "waiting_for_dp_proc"
                    )

                    ## trigger allocation
                    self.task_state[msg["task_id"]][
                        "handler_proc_dp"
                    ] = self.env.process(self.task_dp_handler(msg["task_id"]))
                    self.task_state[msg["task_id"]][
                        "handler_proc_resource"
                    ] = self.env.process(self.task_resources_handler(msg["task_id"]))

                    self.task_state[msg["task_id"]][
                        "blk2accum_getters"
                    ] = dict()  # blk_idx: getter

                    msg['task_init_event'].succeed()

    def _handle_accum_block_waiters(self, task_id):
        this_task = self.task_state[task_id]
        resource_demand = this_task["resource_request"]
        dp_committed_event = this_task["dp_committed_event"]

        wait_for_all_getter_proc = self.env.all_of(
            list(this_task["blk2accum_getters"].values())
        )

        try:
            if self.env.config['task.timeout.enabled']:
                timeout_evt = self.env.timeout(
                    self.env.config['task.timeout.interval'], TIMEOUT_VAL
                )
                permitted_or_timeout_val = yield wait_for_all_getter_proc | timeout_evt

            else:
                permitted_or_timeout_val = yield wait_for_all_getter_proc

            if wait_for_all_getter_proc.triggered:
                self.debug(task_id, "get all dp from blocks")
                return 0

            else:
                assert TIMEOUT_VAL in list(permitted_or_timeout_val.values())
                raise DprequestTimeoutError()

        except (
                StopReleaseDpError,
                InsufficientDpException,
                DprequestTimeoutError,) as err:
            self.debug(
                task_id,
                "policy=%s, fail to acquire dp due to" % self.dp_policy,
                err.__repr__(),
            )
            # interrupt dp_waiting_proc
            if this_task["handler_proc_resource"].is_alive:
                this_task["handler_proc_resource"].interrupt(
                    DpHandlerMessageType.DP_HANDLER_INTERRUPT_MSG
                )
            dp_committed_event.fail(err)
            removed_accum_cn = []
            missing_waiting_accum_cn = []
            fullfilled_blk = []
            unfullfilled_blk = []
            for blk_idx, get_event in this_task[
                "blk2accum_getters"
            ].items():  # get_evt_block_mapping.items():
                if get_event.triggered and get_event.ok:
                    fullfilled_blk.append(blk_idx)
                elif (not get_event.triggered) or (not get_event.ok):
                    unfullfilled_blk.append(blk_idx)
                    get_event.cancel()  # if not triggered pop from waiters
                    get_event.defused = True

                this_block = self.block_dp_storage.items[blk_idx]
                dp_container = this_block["dp_container"]
                if task_id in this_block['waiting_tid2accum_containers']:
                    this_block['waiting_tid2accum_containers'].pop(task_id)
                    removed_accum_cn.append(task_id)
                else:
                    missing_waiting_accum_cn.append(blk_idx)
                if get_event.triggered and get_event.ok:

                    assert (
                                dp_container.level + get_event.amount
                                < dp_container.capacity
                                + self.env.config['sim.numerical_delta']
                        )

            if len(removed_accum_cn) != 0:
                self.debug(
                    task_id,
                    "accum containers removed by task handler for blocks %s"
                    % removed_accum_cn,
                )

            if len(missing_waiting_accum_cn) != 0:
                self.debug(
                    task_id,
                    "accum containers removed by sched for blocks %s"
                    % removed_accum_cn,
                )

            self.debug(task_id, "fullfilled block demand getter: %s" % fullfilled_blk)
            self.debug(
                task_id, "unfullfilled block demand getter: %s" % unfullfilled_blk
            )

            return 1


    def _check_task_admission_control(self, task_id):

        this_task = self.task_state[task_id]
        resource_demand = this_task["resource_request"]
        dp_committed_event = this_task["dp_committed_event"]
        if not self.is_rdp:
            # only check uncommitted dp capacity
            # peek remaining DP, reject if DP is already insufficient
            for i in resource_demand["block_idx"]:
                this_block = self.block_dp_storage.items[i]
                capacity = this_block["dp_container"].capacity
                if (
                        capacity + self.env.config['sim.numerical_delta']
                        < resource_demand["epsilon"]
                ):
                    self.debug(
                        task_id,
                        "DP is insufficient before asking dp scheduler, Block ID: %d, remain epsilon: %.3f"
                        % (i, capacity),
                    )
                    if this_task["handler_proc_resource"].is_alive:
                        this_task["handler_proc_resource"].interrupt(
                            DpHandlerMessageType.DP_HANDLER_INTERRUPT_MSG
                        )
                    # inform user's dp waiting task
                    dp_committed_event.fail(
                        InsufficientDpException(
                            "DP request is rejected by handler admission control, Block ID: %d, remain epsilon: %.3f"
                            % (i, capacity)
                        )
                    )
                    return False
                elif self.is_accum_container_sched and (
                        not this_block['block_proc'].is_alive
                ):
                    dp_committed_event.fail(
                        InsufficientDpException(
                            "DP request is rejected by handler admission control, Block %d sched is inactive"
                            % i
                        )
                    )
                    return False
        else:
            for b in resource_demand["block_idx"]:
                for j, e in enumerate(resource_demand["e_rdp"]):
                    if (
                            self.block_dp_storage.items[b]["rdp_budget_curve"][j]
                            - self.block_dp_storage.items[b]["rdp_consumption"][j]
                            >= e
                    ):
                        break
                else:
                    self.debug(
                        task_id,
                        "RDP is insufficient before asking rdp scheduler, Block ID: %d"
                        % (b),
                    )
                    if this_task["handler_proc_resource"].is_alive:
                        this_task["handler_proc_resource"].interrupt(
                            DpHandlerMessageType.DP_HANDLER_INTERRUPT_MSG
                        )
                    # inform user's dp waiting task
                    this_task['is_dp_granted'] = False
                    dp_committed_event.fail(
                        InsufficientDpException(
                            "RDP request is rejected by handler admission control, Block ID: %d "
                            % (b)
                        )
                    )
                    return False
        return True

    def _do_fcfs(self, task_id):
        this_task = self.task_state[task_id]
        resource_demand = this_task["resource_request"]
        dp_committed_event = this_task["dp_committed_event"]
        if not self.is_rdp:
            for i in resource_demand["block_idx"]:
                # after admission control check, only need to handle numerical accuracy
                self.block_dp_storage.items[i]["dp_container"].get(
                    min(
                        resource_demand["epsilon"],
                        self.block_dp_storage.items[i]["dp_container"].level,
                    )
                )
        else:
            self._commit_rdp_allocation(
                resource_demand["block_idx"], resource_demand["e_rdp"]
            )
            this_task["dp_committed_event"].succeed()

    def _rdp_update_quota_balance(self, blk_idx):
        this_block = self.block_dp_storage.items[blk_idx]
        this_block['rdp_quota_balance'] = list(
            map(
                lambda d: d[0] - d[1],
                zip(this_block['rdp_quota_curve'], this_block['rdp_consumption']),
            )
        )

    def _check_n_update_block_state(self, task_id):
        this_task = self.task_state[task_id]
        resource_demand = this_task["resource_request"]
        blocks_retired_by_this_task = []
        dp_committed_event = this_task["dp_committed_event"]
        # quota increment
        quota_increment_idx = []
        retired_blocks_before_arrival = []
        inactive_block_sched_procs = []
        for i in resource_demand["block_idx"]:
            this_block = self.block_dp_storage.items[i]
            this_block["arrived_task_num"] += 1

            if this_block["retire_event"].triggered:
                assert this_block["retire_event"].ok
                retired_blocks_before_arrival.append(i)
                if self.does_task_handler_unlock_quota:
                    continue

            # unlock quota by task
            elif self.does_task_handler_unlock_quota:
                new_quota_unlocked = None

            # update quota
            if not self.is_rdp:
                if self.is_dp_policy_dpfn:
                    assert not this_block["retire_event"].triggered
                    quota_increment = self.env.config["resource_master.block.init_epsilon"] / self.denom

                    assert (
                            quota_increment
                            < this_block['dp_container'].level
                            + self.env.config['sim.numerical_delta']
                    )
                    if (
                            -self.env.config['sim.numerical_delta']
                            < quota_increment - this_block['dp_container'].level
                            < self.env.config['sim.numerical_delta']
                    ):
                        get_amount = this_block['dp_container'].level
                    else:
                        get_amount = quota_increment

                    this_block['dp_container'].get(get_amount)
                    block_quota = this_block['dp_quota']
                    assert -self.env.config['sim.numerical_delta'] < block_quota.capacity - block_quota.level - get_amount
                    block_quota.put(min(get_amount,block_quota.capacity - block_quota.level))
                    new_quota_unlocked = True

                elif self.is_dp_policy_dpft:
                    pass
                elif self.is_dp_policy_dpfna:
                    assert not this_block["retire_event"].triggered
                    age = self.env.now - this_block["create_time"]

                    x = age / self.env.config['resource_master.block.lifetime']
                    target_quota = ((x - 1) / (-0.0 * x + 1) + 1) * self.env.config[
                        "resource_master.block.init_epsilon"
                    ]
                    released_quota = (
                            self.env.config["resource_master.block.init_epsilon"]
                            - this_block['dp_container'].level
                    )
                    get_amount = target_quota - released_quota
                    this_block['dp_container'].get(get_amount)
                    this_block['dp_quota'].put(get_amount)
                    new_quota_unlocked = True

                elif self.is_dp_policy_rr_n2:
                    if not this_block[
                        "retire_event"
                    ].triggered:  # sched  finished wont be allocated
                        # centralized update;
                        quota_increment = (
                                self.env.config["resource_master.block.init_epsilon"]
                                / self.denom
                        )
                        assert this_block["arrived_task_num"] <= self.denom
                        assert (
                                quota_increment
                                < this_block['dp_container'].level
                                + self.env.config['sim.numerical_delta']
                        )
                        if (
                                -self.env.config['sim.numerical_delta']
                                < quota_increment - this_block['dp_container'].level
                                < self.env.config['sim.numerical_delta']
                        ):
                            get_amount = this_block['dp_container'].level
                        else:
                            get_amount = quota_increment
                        this_block['dp_container'].get(get_amount)

                        this_block["residual_dp"] = (
                                this_block["global_epsilon_dp"]
                                * this_block["arrived_task_num"]
                                / self.denom
                        )
                        new_quota_unlocked = True
                    else:
                        assert not this_block['block_proc'].is_alive
                        inactive_block_sched_procs.append(i)
                        continue

                elif self.is_dp_policy_rr_t:
                    if not this_block['block_proc'].is_alive:
                        inactive_block_sched_procs.append(i)
                        continue
                elif self.is_dp_policy_rr_n:
                    if this_block['block_proc'].is_alive:
                        this_block['_mail_box'].put(task_id)
                    else:
                        inactive_block_sched_procs.append(i)
                        continue
                else:
                    raise NotImplementedError()

            else:
                assert self.is_rdp
                if self.is_dp_policy_dpfn:
                    assert not this_block["retire_event"].triggered
                    fraction = this_block["arrived_task_num"] / self.denom
                    if fraction < 1:
                        this_block['rdp_quota_curve'] = [
                            bjt * fraction for bjt in this_block['rdp_budget_curve']
                        ]
                    else:
                        this_block['rdp_quota_curve'] = this_block[
                            'rdp_budget_curve'
                        ].copy()

                    new_quota_unlocked = True

                elif self.is_dp_policy_dpft:
                    pass

                else:
                    raise NotImplementedError()

            if self.does_task_handler_unlock_quota and new_quota_unlocked:
                quota_increment_idx.append(i)
                if self.is_rdp:
                    self._rdp_update_quota_balance(i)

            if self.is_N_based_retire and (
                    this_block["arrived_task_num"] == self.denom
            ):
                this_block["retire_event"].succeed()
                self._retired_blocks.add(i)
                blocks_retired_by_this_task.append(i)

        # for RR policy, add accum container and getter per task
        if self.is_accum_container_sched:
            if (len(inactive_block_sched_procs) == 0 ):  # make sure no retired blocks before
                if not self.is_rdp:
                    for i in resource_demand["block_idx"]:
                        # need to wait to get per-task accum containers
                        this_block = self.block_dp_storage.items[i]
                        accum_cn = DummyPutPool(
                            self.env, capacity=resource_demand["epsilon"], init=0.0
                        )
                        # will disable get event when when fail to grant.
                        this_task["blk2accum_getters"][i] = accum_cn.get(
                            resource_demand["epsilon"]
                            * (1 - self.env.config['sim.numerical_delta'])
                        )
                        this_block["waiting_tid2accum_containers"][task_id] = accum_cn
                    self.debug(
                        "waiting_tid2accum_containers:  %d,  %s"
                        % (task_id, resource_demand["block_idx"])
                    )
                else:
                    raise NotImplementedError('no rdp x RR')
            else:
                assert this_task["handler_proc_resource"].is_alive
                this_task["handler_proc_resource"].interrupt(
                    DpHandlerMessageType.DP_HANDLER_INTERRUPT_MSG
                )
                assert not dp_committed_event.triggered

                dp_committed_event.fail(
                    StopReleaseDpError(
                        "bbb task %d rejected dp, due to block %s has retired for RR"
                        % (task_id, retired_blocks_before_arrival)
                    )
                )
                # assert not dp_committed_event.ok
        if IS_DEBUG and len(blocks_retired_by_this_task):
            self.debug(
                task_id,
                'blocks No. %s get retired.' % blocks_retired_by_this_task.__repr__(),
            )
        if len(quota_increment_idx) != 0:
            self.dp_sched_mail_box.put(quota_increment_idx)
        # for RR fail tasks when block is retired.
        if dp_committed_event.triggered:
            assert not dp_committed_event.ok
            return 1

    def _handle_quota_sched_permission(self, task_id):
        this_task = self.task_state[task_id]
        resource_demand = this_task["resource_request"]
        dp_committed_event = this_task["dp_committed_event"]

        def wait_for_permit():

            yield self.task_state[task_id]["dp_permitted_event"]

        wait_for_permit_proc = self.env.process(wait_for_permit())
        try:
            t0 = self.env.now
            if self.env.config['task.timeout.enabled']:
                permitted_or_timeout_val = (
                    yield wait_for_permit_proc
                          | self.env.timeout(
                        self.env.config['task.timeout.interval'], TIMEOUT_VAL
                    )
                )
            else:
                permitted_or_timeout_val = yield wait_for_permit_proc

            if wait_for_permit_proc.triggered:
                self.debug(
                    task_id,
                    "grant_dp_permitted after ",
                    timedelta(seconds=(self.env.now - t0)),
                )
                if not self.is_rdp:
                    for i in resource_demand["block_idx"]:
                        this_block = self.block_dp_storage.items[i]

                        this_block["dp_quota"].get(
                            min(
                                resource_demand["epsilon"], this_block["dp_quota"].level
                            )
                        )
                    return 0
                else:
                    pass  # already increase rdp quota while scheduling.
            else:
                assert list(permitted_or_timeout_val.values())[0] == TIMEOUT_VAL
                raise DprequestTimeoutError()

        except (DprequestTimeoutError, DpBlockRetiredError) as err:
            if isinstance(err, DprequestTimeoutError):
                del_idx = self.dp_waiting_tasks.items.index(task_id)

                self.debug(
                    task_id,
                    "dp request timeout after %d "
                    % self.env.config['task.timeout.interval'],
                )
                self.debug(task_id, "task get dequeued from wait queue")
                del self.dp_waiting_tasks.items[del_idx]

            self.debug(
                task_id,
                "policy=%s, fail to acquire dp: %s" % (self.dp_policy, err.__repr__()),
            )

            if isinstance(err, DpBlockRetiredError):
                # should not issue get to quota
                assert not self.task_state[task_id]["dp_permitted_event"].ok

            # interrupt dp_waiting_proc
            if this_task["handler_proc_resource"].is_alive:
                this_task["handler_proc_resource"].interrupt(
                    DpHandlerMessageType.DP_HANDLER_INTERRUPT_MSG
                )

            dp_committed_event.fail(err)

            return 1

    # should trigger committed dp_committed_event after return
    def task_dp_handler(self, task_id):
        self.debug(task_id, "Task DP handler created")
        this_task = self.task_state[task_id]
        dp_committed_event = this_task["dp_committed_event"]

        resource_demand = this_task["resource_request"]
        # admission control

        # getevent -> blk_idx

        if self.is_dp_policy_fcfs:
            this_task["is_admission_control_ok"] = self._check_task_admission_control(
                task_id
            )
            if not this_task["is_admission_control_ok"]:
                return
            self._do_fcfs(task_id)
            # always grant because admission control is ok
            self.task_state[task_id]["is_dp_granted"] = True

        else:  # policy other than fcfs
            if self.is_admission_control_enabled:
                this_task[
                    "is_admission_control_ok"
                ] = self._check_task_admission_control(task_id)
                if not this_task["is_admission_control_ok"]:
                    return
            update_status = self._check_n_update_block_state(task_id)
            if update_status == 1:
                self.task_state[task_id]["is_dp_granted"] = False
            else:
                if self.is_centralized_quota_sched:
                    # dp_policy_dpft only needs enqueue
                    self.dp_waiting_tasks.put(task_id)
                    self.dp_sched_mail_box.put(task_id)
                    a = yield from self._handle_quota_sched_permission(task_id)
                    if a == 0:
                        self.task_state[task_id]["is_dp_granted"] = True

                else:  ## RR-t RR-n
                    a = yield from self._handle_accum_block_waiters(task_id)
                    if a == 0:
                        self.task_state[task_id]["is_dp_granted"] = True
                # commit dp after permission from sched or directly for fcfs.

        if self.task_state[task_id]["is_dp_granted"]:

            if not self.is_rdp:
                assert not dp_committed_event.triggered
                self._commit_dp_allocation(
                    resource_demand["block_idx"], epsilon=resource_demand["epsilon"]
                )
                dp_committed_event.succeed()
            else:
                assert dp_committed_event.ok
                # rdp is already committed, and triggered in scheduling algo
                pass

        else:
            assert not dp_committed_event.ok

    def task_resources_handler(self, task_id):
        self.debug(task_id, "Task resource handler created")
        # add to resources wait queue
        self.resource_waiting_tasks.put(task_id)
        self.resource_sched_mail_box.put(
            {"msg_type": ResourceHandlerMessageType.RESRC_TASK_ARRIVAL, "task_id": task_id}
        )
        this_task = self.task_state[task_id]
        resource_allocated_event = this_task["resource_allocated_event"]
        resource_demand = this_task["resource_request"]
        resource_permitted_event = this_task["resource_permitted_event"]

        success_resrc_get_events = []

        try:

            yield resource_permitted_event
            get_cpu_event = self.cpu_pool.get(resource_demand["cpu"])
            success_resrc_get_events.append(get_cpu_event)

            if not self.is_cpu_needed_only:
                get_memory_event = self.memory_pool.get(resource_demand["memory"])
                success_resrc_get_events.append(get_memory_event)

                get_gpu_event = self.gpu_pool.get(resource_demand["gpu"])
                success_resrc_get_events.append(get_gpu_event)

            resource_allocated_event.succeed()
            self.task_state[task_id]["resource_allocate_timestamp"] = self.env.now

        # todo, maybe add another exception handling chain: interrupt dp handler....
        except RejectResourcePermissionError as err:
            # sched find task's dp is rejected, then fail its resource handler.
            resource_allocated_event.fail(ResourceAllocFail(err))

            return

        except simpy.Interrupt as err:
            assert err.args[0] == DpHandlerMessageType.DP_HANDLER_INTERRUPT_MSG
            assert len(success_resrc_get_events) == 0
            resource_allocated_event.fail(
                ResourceAllocFail("Abort resource request: %s" % err)
            )
            defuse(resource_permitted_event)
            # interrupted while permitted
            # very likely
            if not resource_permitted_event.triggered:
                assert task_id in self.resource_waiting_tasks.items
                self.resource_waiting_tasks.get(filter=lambda x: x == task_id)

                for i in self.resource_sched_mail_box.items:
                    if (i['task_id'] == task_id) and (
                            i['msg_type'] == ResourceHandlerMessageType.RESRC_TASK_ARRIVAL
                    ):
                        self.resource_sched_mail_box.get(
                            filter=lambda x: (x['task_id'] == task_id)
                                             and (x['msg_type'] == ResourceHandlerMessageType.RESRC_TASK_ARRIVAL)
                        )
                pass
            # fixme coverage
            else:
                assert resource_permitted_event.ok and (
                    not resource_permitted_event.processed
                )
                self.debug(
                    task_id,
                    "warning: resource permitted but abort to allocate due to interrupt",
                )
                self.resource_sched_mail_box.put(
                    {"msg_type": ResourceHandlerMessageType.RESRC_PERMITED_FAIL_TO_ALLOC, "task_id": task_id}
                )

            return

        exec_proc = this_task['execution_proc']
        try:
            # yield task_completion_event
            yield exec_proc

        except simpy.Interrupt as err:
            assert err.args[0] == DpHandlerMessageType.DP_HANDLER_INTERRUPT_MSG
            if exec_proc.is_alive:
                defuse(exec_proc)
                exec_proc.interrupt(DpHandlerMessageType.DP_HANDLER_INTERRUPT_MSG)
                v = yield exec_proc | self.env.timeout(
                    self.env.config['sim.instant_timeout'], TIMEOUT_VAL
                )
                # exec_proc should exit immeidately after interrupt
                assert v != TIMEOUT_VAL

        self.cpu_pool.put(resource_demand["cpu"])
        if not self.is_cpu_needed_only:
            self.gpu_pool.put(resource_demand["gpu"])
            self.memory_pool.put(resource_demand["memory"])

        self.debug(task_id, "Resource released")
        self.resource_sched_mail_box.put(
            {"msg_type": ResourceHandlerMessageType.RESRC_RELEASE, "task_id": task_id}
        )

        return

    ##  unlock dp quota for dpft policy
    def _dpft_subloop_unlock_quota(self, block_id):
        self.debug(
            'block_id %d release quota subloop start at %.3f' % (block_id, self.env.now)
        )
        this_block = self.block_dp_storage.items[block_id]
        # wait first to sync clock
        yield self.global_clock.next_tick
        if this_block["end_of_life"] <= self.env.now:
            if not self.is_rdp:
                total_dp = this_block["dp_container"].level
                this_block["dp_container"].get(total_dp)
                this_block["dp_quota"].put(total_dp)

            else:
                this_block['rdp_quota_curve'] = this_block['rdp_budget_curve'].copy()
                self._rdp_update_quota_balance(block_id)

            self.dp_sched_mail_box.put([block_id])
            this_block["retire_event"].succeed()
            self._retired_blocks.add(block_id)
            return

        total_ticks = (
                int(
                    (this_block["end_of_life"] - self.env.now)
                    / self.global_clock.seconds_per_tick
                )
                + 1
        )
        init_level = this_block["dp_container"].level

        for t in range(total_ticks):
            if not self.is_rdp:
                should_release_amount = init_level / total_ticks * (t + 1)
                get_amount = should_release_amount - (
                        init_level - this_block["dp_container"].level
                )

                if t + 1 == total_ticks:
                    assert (
                            -self.env.config['sim.numerical_delta']
                            < this_block["dp_container"].level - get_amount
                            < self.env.config['sim.numerical_delta']
                    )
                    get_amount = this_block["dp_container"].level

                this_block["dp_container"].get(get_amount)

                assert (
                        this_block["dp_quota"].level + get_amount
                        < this_block["dp_quota"].capacity
                        + self.env.config['sim.numerical_delta']
                )
                put_amount = min(
                    get_amount,
                    this_block["dp_quota"].capacity - this_block["dp_quota"].level,
                )
                this_block["dp_quota"].put(put_amount)
            else:
                this_block['rdp_quota_curve'] = [
                    b * ((t + 1) / total_ticks) for b in this_block['rdp_budget_curve']
                ]
                self._rdp_update_quota_balance(block_id)
            self.debug(
                'block_id %d release %.3f fraction at %.3f'
                % (block_id, (t + 1) / total_ticks, self.env.now)
            )
            self.dp_sched_mail_box.put([block_id])
            if t + 1 != total_ticks:
                yield self.global_clock.next_tick
            else:
                break

        if not self.is_rdp:
            assert this_block["dp_container"].level == 0

        this_block["retire_event"].succeed()
        self._retired_blocks.add(block_id)
        # HACK quotad increment msg, to trigger waiting task exceptions
        self.dp_sched_mail_box.put([block_id])
        self.debug('block_id %d retired with 0 DP left' % block_id)

    def _dpfn_subloop_eol_retire(self, block_id):
        this_block = self.block_dp_storage.items[block_id]
        lifetime_ub = (
                self.env.config['resource_master.block.arrival_interval']
                * self.env.config['task.demand.num_blocks.elephant']
                * 1.01
        )
        yield self.env.timeout(lifetime_ub)
        if not this_block['retire_event'].triggered:
            this_block['retire_event'].succeed()
            self.debug(
                'DPF-N,N=%d, block %d get retired by timeout with %d arrived tasks'
                % (self.denom, block_id, this_block['arrived_task_num'])
            )

    def _rr_n_subloop_eol_sched(self, block_id):
        this_block = self.block_dp_storage.items[block_id]

        # due to discretization, shift release during tick seceonds period to the start of this second.
        # therefore, last release should happen before end of life
        yield self.env.timeout(self.env.config['resource_master.block.lifetime'])
        # allocate in ascending order
        waiting_task_cn_mapping = this_block["waiting_tid2accum_containers"]
        if len(waiting_task_cn_mapping) > 0:
            total_alloc = 0  # this is inaccurate
            is_dp_sufficient = True
            unlocked_dp_quota_total = this_block["residual_dp"]
            waiting_tasks_asc = list(
                sorted(waiting_task_cn_mapping, key=lambda x: x[1].capacity)
            )
            self.debug( 'block %d EOL, remove all waiting accum containers %s'
                % (block_id, waiting_tasks_asc)
            )
            for task_id in waiting_tasks_asc:
                cn = waiting_task_cn_mapping[task_id]
                if (
                        total_alloc + cn.capacity
                        <= unlocked_dp_quota_total + self.env.config['sim.numerical_delta']
                ):

                    cn.put(cn.capacity)
                    total_alloc = total_alloc + cn.capacity
                elif is_dp_sufficient:

                    waiting_evt = cn._get_waiters.pop(0)
                    waiting_evt.fail(
                        InsufficientDpException(
                            "block %d remaining uncommitted DP is insufficient for remaining ungranted dp of task %d"
                            % (block_id, task_id)
                        )
                    )

                    is_dp_sufficient = False
                else:
                    assert not is_dp_sufficient

                    waiting_evt = cn._get_waiters.pop(0)
                    waiting_evt.fail(
                        InsufficientDpException(
                            "block %d remaining uncommitted DP is insufficient for remaining ungranted dp of task %d"
                            % (block_id, task_id)
                        )
                    )
                # remove all waiting accum cn at the eol
                this_block["waiting_tid2accum_containers"].pop(task_id)

        # fail others waiting tasks

    ## for RR N policy
    def _rr_nn_subloop_unlock_quota_n_sched(self, block_id):

        this_block = self.block_dp_storage.items[block_id]
        this_block["residual_dp"] = 0.0

        init_level = this_block["dp_container"].level

        while True:

            new_tid = yield this_block['_mail_box'].get()
            n = this_block['arrived_task_num']  # should update in task hander
            if n == self.denom:
                # already retired by task handler
                assert this_block["retire_event"].ok
            elif (
                    n > self.denom
                    and this_block["residual_dp"] < self.env.config['sim.numerical_delta']
            ):
                break

            get_amount = 0  # t + 1 > total_ticks
            if n < self.denom:
                should_release_amount = init_level / self.denom * n
                get_amount = should_release_amount - (
                        init_level - this_block["dp_container"].level
                )

            elif n == self.denom:
                get_amount = this_block["dp_container"].level

            if get_amount != 0:
                this_block["dp_container"].get(get_amount)
                this_block["residual_dp"] += get_amount
            else:
                assert this_block["dp_container"].level == 0

            # only allocate among active getter tasks
            waiting_task_cn_mapping = this_block["waiting_tid2accum_containers"]


            if len(waiting_task_cn_mapping) > 0:

                desired_dp = {
                    tid: cn.capacity - cn.level
                    for tid, cn in waiting_task_cn_mapping.items()
                }
                # self.debug(block_id, "call max_min_fair_allocation")
                fair_allocation = max_min_fair_allocation(
                    demand=desired_dp, capacity=this_block["residual_dp"]
                )

                # all waiting task is granted by this block, return back unused dp
                if sum(fair_allocation.values()) < this_block["residual_dp"]:
                    this_block["residual_dp"] -= sum(
                        fair_allocation.values()
                    )

                else:
                    this_block["residual_dp"] = 0.0
                    yield self.env.timeout(delay=0)  # may update residual ??????

                for tid, dp_alloc_amount in fair_allocation.items():
                    cn = this_block["waiting_tid2accum_containers"][tid]

                    if (
                            -self.env.config['sim.numerical_delta']
                            < (cn.capacity - cn.level) - dp_alloc_amount
                            < self.env.config['sim.numerical_delta']
                    ):
                        dp_alloc_amount = cn.capacity - cn.level
                        cn.put(dp_alloc_amount)
                        this_block["waiting_tid2accum_containers"].pop(tid)
                        self.debug(
                            tid,
                            "accum containers granted and removed for block %s"
                            % block_id,
                        )
                    else:
                        cn.put(dp_alloc_amount)

        # now >= end of life
        # wait for dp getter event processed, reject untriggered get
        yield self.env.timeout(delay=0)
        rej_waiting_task_cn_mapping = this_block["waiting_tid2accum_containers"]

        if len(rej_waiting_task_cn_mapping) != 0:
            self.debug(
                "block %d last period of lifetime, with waiting tasks: " % block_id,
                list(rej_waiting_task_cn_mapping.keys()),
            )
            for task_id in list(rej_waiting_task_cn_mapping.keys()):
                cn = rej_waiting_task_cn_mapping[task_id]
                # avoid getter triggered by cn
                waiting_evt = cn._get_waiters.pop(0)
                waiting_evt.fail(
                    StopReleaseDpError(
                        "task %d rejected dp, due to block %d has stopped release"
                        % (task_id, block_id)
                    )
                )
                this_block["waiting_tid2accum_containers"].pop(task_id)
        else:
            self.debug("block %d out of dp, with NO waiting task" % block_id)

        return 0

    ## for RR T policy
    def _rr_t_subloop_unlock_quota_n_sched(self, block_id):

        this_block = self.block_dp_storage.items[block_id]
        this_block["residual_dp"] = 0.0

        # due to discretization, shift release during tick seceonds period to the start of this second.
        # therefore, last release should happen before end of life
        # wait first to sync clock
        yield self.global_clock.next_tick
        t0 = self.env.now
        total_ticks = (
                int((this_block["end_of_life"] - t0) / self.global_clock.seconds_per_tick)
                + 1
        )
        init_level = this_block["dp_container"].level
        ticker_counter = count()
        while True:
            t = next(ticker_counter) + 1
            get_amount = 0  # t + 1 > total_ticks
            if t < total_ticks:
                should_release_amount = init_level / total_ticks * t
                get_amount = should_release_amount - (
                        init_level - this_block["dp_container"].level
                )

            elif t == total_ticks:
                get_amount = this_block["dp_container"].level
                this_block["retire_event"].succeed()

            if get_amount != 0:
                this_block["dp_container"].get(get_amount)
                this_block["residual_dp"] = this_block["residual_dp"] + get_amount
            else:
                assert this_block["dp_container"].level == 0

            # only allocate among active getter tasks
            waiting_task_cn_mapping = this_block["waiting_tid2accum_containers"]

            if len(waiting_task_cn_mapping) > 0:

                desired_dp = {
                    tid: cn.capacity - cn.level
                    for tid, cn in waiting_task_cn_mapping.items()
                }
                # self.debug(block_id, "call max_min_fair_allocation")
                fair_allocation = max_min_fair_allocation(
                    demand=desired_dp, capacity=this_block["residual_dp"]
                )

                # all waiting task is granted by this block, return back unused dp
                if sum(fair_allocation.values()) < this_block["residual_dp"]:
                    this_block["residual_dp"] = this_block["residual_dp"] - sum(
                        fair_allocation.values()
                    )

                else:
                    this_block["residual_dp"] = 0.0

                for tid, dp_alloc_amount in fair_allocation.items():
                    cn = this_block["waiting_tid2accum_containers"][tid]
                    if (
                            -self.env.config['sim.numerical_delta']
                            < (cn.capacity - cn.level) - dp_alloc_amount
                            < self.env.config['sim.numerical_delta']
                    ):
                        dp_alloc_amount = cn.capacity - cn.level
                        cn.put(dp_alloc_amount)
                        this_block["waiting_tid2accum_containers"].pop(tid)
                        self.debug(
                            tid,
                            "accum containers granted and removed for block %s"
                            % block_id,
                        )
                    else:
                        cn.put(dp_alloc_amount)

            if (
                    this_block["residual_dp"] < self.env.config['sim.numerical_delta']
                    and t >= total_ticks
            ):
                break
            else:
                yield self.global_clock.next_tick
        # now >= end of life
        # wait for dp getter event processed, reject untriggered get
        yield self.env.timeout(delay=0)
        rej_waiting_task_cn_mapping = this_block[
            "waiting_tid2accum_containers"
        ]

        if len(rej_waiting_task_cn_mapping) != 0:
            self.debug(
                "block %d EOL, removing accum containers for waiting tasks: %s"
                % (block_id, list(rej_waiting_task_cn_mapping.keys()))
            )
            for task_id in list(rej_waiting_task_cn_mapping.keys()):
                cn = rej_waiting_task_cn_mapping[task_id]
                # avoid getter triggered by cn
                waiting_evt = cn._get_waiters.pop(0)
                waiting_evt.fail(
                    StopReleaseDpError(
                        "aaa task %d rejected dp, due to block %d has run out of dp"
                        % (task_id, block_id)
                    )
                )
                this_block["waiting_tid2accum_containers"].pop(task_id)
        else:
            self.debug("block %d run out of dp, with NO waiting task" % block_id)

        return 0

    def _dp_dpfna_eol_callback_gen(self, b_idx):
        cn = self.block_dp_storage.items[b_idx]['dp_container']
        quota = self.block_dp_storage.items[b_idx]['dp_quota']

        def eol_callback(eol_evt):
            lvl = cn.level
            cn.get(lvl)
            assert (
                    quota.level + lvl
                    < quota.capacity + self.env.config['sim.numerical_delta']
            )
            lvl = min(lvl, quota.capacity - quota.level)
            quota.put(lvl)
            assert cn.level == 0
            # inform mail box retire
            self.dp_sched_mail_box.put([b_idx])
            self.block_dp_storage.items[b_idx]["retire_event"].succeed()
            self._retired_blocks.add(b_idx)
            self.debug(
                'block %d EOF, move remaining dp from container to quota' % b_idx
            )

        return eol_callback

    def generate_datablocks_loop(self):
        cur_block_nr = 0
        block_id = count()
        is_static_blocks = self.env.config["resource_master.block.is_static"]
        init_amount = self.env.config["resource_master.block.init_amount"]
        while True:
            if cur_block_nr > init_amount:
                yield self.env.timeout(
                    self.env.config["resource_master.block.arrival_interval"]
                )

            elif cur_block_nr < init_amount:
                cur_block_nr += 1

            elif cur_block_nr == init_amount:
                cur_block_nr += 1
                self.init_blocks_ready.succeed()
                if is_static_blocks:
                    self.debug(
                        'epsilon initial static data blocks: %s'
                        % pp.pformat(
                            [
                                blk['dp_container'].capacity
                                for blk in self.block_dp_storage.items
                            ]
                        )
                    )
                    return

            # generate block_id
            cur_block_id = next(block_id)
            total_dp = self.env.config['resource_master.block.init_epsilon']
            new_block = DummyPool(
                self.env,
                capacity=total_dp,
                init=total_dp,
                name=cur_block_id,
                hard_cap=True,
            )
            rdp_budget_curve = []
            if self.is_rdp:
                for a in sorted(ALPHAS):
                    assert a > 1
                    total_delta = self.env.config['resource_master.block.init_delta']
                    total_rdp = max(0, total_dp - math.log(1 / total_delta) / (a - 1))
                    rdp_budget_curve.append(
                        total_rdp
                    )

            if self.is_T_based_retire:
                EOL = self.env.now + self.env.config['resource_master.block.lifetime']
            else:
                EOL = None
            # no sched fcfs
            accum_cn_dict = None
            new_quota = None
            if self.is_accum_container_sched:
                accum_cn_dict = dict()
                new_quota = None
            elif self.is_centralized_quota_sched:
                accum_cn_dict = None
                new_quota = DummyPool(
                    self.env,
                    capacity=total_dp,
                    init=0,
                    name=cur_block_id,
                    hard_cap=True,
                )

            block_item = {
                "global_epsilon_dp": total_dp,
                "dp_container": new_block,
                "rdp_budget_curve": rdp_budget_curve,
                "rdp_quota_curve": [0.0, ] * len(rdp_budget_curve),
                "rdp_consumption": [0.0, ] * len(rdp_budget_curve),
                "rdp_quota_balance": [0.0, ]
                                     * len(rdp_budget_curve),  # balance = quota - consumption
                "dp_quota": new_quota,  # for dpf policy
                # lifetime is # of periods from born to end
                "end_of_life": EOL,
                "waiting_tid2accum_containers": accum_cn_dict,  # task_id: container, for rate limiting policy
                "retire_event": self.env.event(),
                'arrived_task_num': 0,
                'last_task_arrival_time': None,
                'create_time': self.env.now,
                'residual_dp': 0.0,
                'block_proc': None,
                '_mail_box': DummyPutQueue(self.env, capacity=1, hard_cap=True),
            }

            self.block_dp_storage.put(block_item)
            self.unused_dp.put(total_dp)
            self.debug("new data block %d created" % cur_block_id)

            if self.is_dp_policy_rr_t:
                if not self.is_rdp:
                    block_item['block_proc'] = self.env.process(
                        self._rr_t_subloop_unlock_quota_n_sched(cur_block_id)
                    )
                else:
                    raise NotImplementedError()

            elif self.is_dp_policy_rr_n2:
                if not self.is_rdp:
                    block_item['block_proc'] = self.env.process(
                        self._rr_n_subloop_eol_sched(cur_block_id)
                    )
                else:
                    raise NotImplementedError()

            elif self.is_dp_policy_dpft:
                # one process for rdp and non-rdp
                block_item['block_proc'] =  self.env.process(self._dpft_subloop_unlock_quota(cur_block_id))

            elif self.is_dp_policy_dpfna:
                block_item['retire_event'] = self.env.timeout(
                    self.env.config['resource_master.block.lifetime']
                )
                block_item['retire_event'].callbacks.append(
                    self._dp_dpfna_eol_callback_gen(
                        self.block_dp_storage.items.index(block_item)
                    )
                )
            elif self.is_dp_policy_rr_n:
                block_item['block_proc'] = self.env.process(
                    self._rr_nn_subloop_unlock_quota_n_sched(cur_block_id)
                )

            elif (
                    self.is_dp_policy_dpfn
                    and not self.env.config['resource_master.block.is_static']
            ):
                block_item['block_proc'] = self.env.process(
                    self._dpfn_subloop_eol_retire(cur_block_id)
                )

    def _commit_dp_allocation(self, block_idx: List[int], epsilon: float):
        """
        each block's capacity is uncommitted DP, commit by deducting capacity by epsilon.
        Args:
            block_idx:
            epsilon:

        Returns:

        """
        assert len(block_idx) > 0
        # verify able to commit
        for i in block_idx:
            this_container = self.block_dp_storage.items[i]["dp_container"]
            assert epsilon <= this_container.capacity or (
                    epsilon - this_container.capacity
                    < self.env.config['sim.numerical_delta']
                    and this_container.level == 0
            )

            assert (
                           this_container.level + epsilon
                   ) < this_container.capacity + self.env.config['sim.numerical_delta']

        for i in block_idx:
            this_container = self.block_dp_storage.items[i]["dp_container"]
            this_container.capacity = max(
                this_container.capacity - epsilon, this_container.level
            )
        committed_amount = min(epsilon * len(block_idx), self.unused_dp.level)
        self.unused_dp.get(committed_amount)
        self.committed_dp.put(committed_amount)

        if IS_DEBUG:
            unused_dp = []
            if (
                    self.is_centralized_quota_sched
            ):  # :self.is_dp_policy_dpfn or self.is_dp_policy_dpft or self.is_dp_policy_dpfa
                for i, block in enumerate(self.block_dp_storage.items):
                    if i in block_idx:
                        unused_dp.append(-round(block['dp_quota'].level, 2))
                    else:
                        unused_dp.append(round(block['dp_quota'].level, 2))
                if self.env.config['workload_test.enabled']:
                    self.debug(
                        "unused dp quota after commit: %s (negative sign denotes committed block)"
                        % pp.pformat(unused_dp)
                    )
            elif self.is_accum_container_sched:
                for i, block in enumerate(self.block_dp_storage.items):
                    # uncommitted dp
                    if i in block_idx:
                        unused_dp.append(-round(block['dp_container'].capacity, 2))
                    else:
                        unused_dp.append(round(block['dp_container'].capacity, 2))
                if self.env.config['workload_test.enabled']:
                    self.debug(
                        "unused dp after commit: %s (negative sign denotes committed block)"
                        % pp.pformat(unused_dp)
                    )

    def get_result_hook(self, result):
        if not self.env.tracemgr.vcd_tracer.enabled:
            return
        cpu_capacity = self.env.config['resource_master.cpu_capacity']
        with open(self.env.config["sim.vcd.dump_file"]) as vcd_file:

            if vcd_file.read(1) == '':
                return

        with open(self.env.config["sim.vcd.dump_file"]) as vcd_file:

            vcd = VcdParser()
            vcd.parse(vcd_file)
            root_data = vcd.scope.toJson()
            assert root_data['children'][0]['children'][2]['name'] == "cpu_pool"
            # at least 11 sample
            if len(root_data['children'][0]['children'][2]['data']) == 0:
                result['CPU_utilization%'] = 0
                return
            elif len(root_data['children'][0]['children'][2]['data']) <= 10:
                self.debug("WARNING: CPU change sample size <= 10")
            idle_cpu_record = map(
                lambda t: (t[0], eval('0' + t[1])),
                root_data['children'][0]['children'][2]['data'],
            )

            idle_cpu_record = list(idle_cpu_record)
            # record should start at time 0
            if idle_cpu_record[0][0] != 0:
                idle_cpu_record = [(0, cpu_capacity)] + idle_cpu_record

            assert (
                    self.env.config['sim.timescale']
                    == self.env.config['sim.duration'].split(' ')[1]
            )
            end_tick = int(self.env.config['sim.duration'].split(' ')[0]) - 1
            if idle_cpu_record[-1][0] != end_tick:
                idle_cpu_record += [
                        (end_tick, idle_cpu_record[-1][1])
                    ]

            t1, t2 = tee(idle_cpu_record)
            next(t2)
            busy_cpu_time = map(
                lambda t: (cpu_capacity - t[0][1]) * (t[1][0] - t[0][0]),
                zip(t1, t2),
            )

        # cal over start and end
        result['CPU_utilization%'] = (
                100 * sum(busy_cpu_time) / (end_tick + 1) / cpu_capacity
        )


class Tasks(Component):
    base_name = 'tasks'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_rand = self.env.rand
        self.add_connections('resource_master')
        self.add_connections('global_clock')

        self.add_process(self.generate_tasks_loop)

        self.task_unpublished_count = DummyPool(self.env)
        self.auto_probe('task_unpublished_count', vcd={})

        self.task_published_count = DummyPool(self.env)
        self.auto_probe('task_published_count', vcd={})

        self.task_sleeping_count = DummyPool(self.env)
        self.auto_probe('task_sleeping_count', vcd={})

        self.task_running_count = DummyPool(self.env)
        self.auto_probe('task_running_count', vcd={})

        self.task_completed_count = DummyPool(self.env)
        self.auto_probe('task_completed_count', vcd={})

        self.task_abort_count = DummyPool(self.env)
        self.auto_probe('task_abort_count', vcd={})

        self.task_ungranted_count = DummyPool(self.env)
        self.auto_probe('task_ungranted_count', vcd={})

        self.task_granted_count = DummyPool(self.env)
        self.auto_probe('task_granted_count', vcd={})

        self.tasks_granted_count_s_dp_s_blk = DummyPool(self.env)
        self.auto_probe('tasks_granted_count_s_dp_s_blk', vcd={})

        self.tasks_granted_count_s_dp_l_blk = DummyPool(self.env)
        self.auto_probe('tasks_granted_count_s_dp_l_blk', vcd={})

        self.tasks_granted_count_l_dp_s_blk = DummyPool(self.env)
        self.auto_probe('tasks_granted_count_l_dp_s_blk', vcd={})

        self.tasks_granted_count_l_dp_l_blk = DummyPool(self.env)
        self.auto_probe('tasks_granted_count_l_dp_l_blk', vcd={})

        self.task_dp_rejected_count = DummyPool(self.env)
        self.auto_probe('task_dp_rejected_count', vcd={})

        if self.env.tracemgr.sqlite_tracer.enabled:
            self.db = self.env.tracemgr.sqlite_tracer.db
            self.db.execute(
                'CREATE TABLE tasks '
                '(task_id INTEGER PRIMARY KEY,'
                ' start_block_id INTEGER,'
                ' num_blocks INTEGER,'
                ' epsilon REAL,'
                ' cpu INTEGER,'
                ' gpu INTEGER,'
                ' memory REAL,'
                ' start_timestamp REAL,'
                ' dp_commit_timestamp REAL,'
                ' resource_allocation_timestamp REAL,'
                ' completion_timestamp REAL,'
                ' publish_timestamp REAL'
                ')'
            )
        else:
            self.db = None

        if self.env.config.get('task.demand.num_cpu.constant') is not None:
            assert isinstance(self.env.config['task.demand.num_cpu.constant'], int)
            self.cpu_dist = lambda: self.env.config['task.demand.num_cpu.constant']
        else:
            num_cpu_min = self.env.config.setdefault('task.demand.num_cpu.min',1)
            num_cpu_max = self.env.config.setdefault('task.demand.num_cpu.max', num_cpu_min)
            self.cpu_dist = partial(
                self.load_rand.randint,
                num_cpu_min,
                num_cpu_max,
            )

        size_memory_min = self.env.config.setdefault('task.demand.size_memory.min',1)
        size_memory_max = self.env.config.setdefault('task.demand.size_memory.max', size_memory_min)
        self.memory_dist = partial( self.load_rand.randint, size_memory_min, size_memory_max )

        num_gpu_min = self.env.config.setdefault('task.demand.num_gpu.min',1)
        num_gpu_max = self.env.config.setdefault('task.demand.num_gpu.max', 1)
        self.gpu_dist = partial( self.load_rand.randint, num_gpu_min, num_gpu_max )


        if self.env.config.get('task.demand.completion_time.constant') is not None:
            self.completion_time_dist = lambda: self.env.config[
                'task.demand.completion_time.constant'
            ]
        else:
            completion_time_min = self.env.config.setdefault('task.demand.completion_time.min', 0)
            completion_time_max = self.env.config.setdefault('task.demand.completion_time.max', completion_time_min)
            self.completion_time_dist = partial(
                self.load_rand.randint, completion_time_min, completion_time_max,)
        choose_one = lambda *kargs, **kwargs: self.load_rand.choices(*kargs, **kwargs)[
            0
        ]
        e_mice_fraction = self.env.config['task.demand.epsilon.mice_percentage'] / 100
        choose_and_discount = lambda *kargs, **kwargs: choose_one(
            *kargs, **kwargs
        ) * self.load_rand.uniform(0.9999999, 1)
        self.epsilon_dist = partial(
            choose_and_discount,
            (
                self.env.config['task.demand.epsilon.mice'],
                self.env.config['task.demand.epsilon.elephant'],
            ),
            (e_mice_fraction, 1 - e_mice_fraction),
        )
        block_mice_fraction = (
                self.env.config['task.demand.num_blocks.mice_percentage'] / 100
        )
        self.num_blocks_dist = partial(
            choose_one,
            (
                self.env.config['task.demand.num_blocks.mice'],
                self.env.config['task.demand.num_blocks.elephant'],
            ),
            (block_mice_fraction, 1 - block_mice_fraction),
        )

    def generate_tasks_loop(self):

        task_id = count()
        arrival_interval_dist = partial(
            self.load_rand.expovariate, 1 / self.env.config['task.arrival_interval']
        )

        ## wait for generating init blocks
        def init_one_task(
                task_id,
                start_block_idx,
                end_block_idx,
                epsilon,
                delta,
                e_rdp,
                completion_time,
                cpu_demand,
                gpu_demand,
                memory_demand,
        ):

            task_process = self.env.process(
                self.task(
                    task_id,
                    start_block_idx,
                    end_block_idx,
                    epsilon,
                    delta,
                    e_rdp,
                    completion_time,
                    cpu_demand,
                    gpu_demand,
                    memory_demand,
                )
            )
            new_task_msg = {
                "message_type": DpHandlerMessageType.NEW_TASK,
                "task_id": task_id,
                "task_process": task_process,
            }

            self.resource_master.mail_box.put(new_task_msg)

        yield self.resource_master.init_blocks_ready
        if not self.env.config['workload_test.enabled']:
            while True:
                yield self.env.timeout(arrival_interval_dist())
                t_id = next(task_id)
                # query existing data blocks
                num_stored_blocks = len(self.resource_master.block_dp_storage.items)
                assert num_stored_blocks > 0
                num_blocks_demand = min(
                    max(1, round(self.num_blocks_dist())), num_stored_blocks
                )
                epsilon = self.epsilon_dist()

                if self.resource_master.is_rdp:
                    sigma = gaussian_dp2sigma(epsilon, 1, DELTA)
                    rdp_demand = compute_rdp_epsilons_gaussian(sigma, ALPHAS)
                else:
                    rdp_demand = None
                init_one_task(
                    task_id = t_id,
                    start_block_idx=num_stored_blocks - num_blocks_demand,
                    end_block_idx=num_stored_blocks - 1,
                    epsilon=epsilon,
                    delta=DELTA,
                    e_rdp=rdp_demand,
                    completion_time=self.completion_time_dist(),
                    cpu_demand=self.cpu_dist(),
                    gpu_demand=self.gpu_dist(),
                    memory_demand=self.memory_dist(),
                )

        else:
            assert self.env.config['workload_test.workload_trace_file']
            with open(self.env.config['workload_test.workload_trace_file']) as f:
                tasks = yaml.load(f, Loader=yaml.FullLoader)
                for t in sorted(tasks, key=lambda x: x['arrival_time']):
                    assert t['arrival_time'] - self.env.now >= 0
                    yield self.env.timeout(t['arrival_time'] - self.env.now)
                    if self.resource_master.is_rdp:
                        sigma = gaussian_dp2sigma(t['epsilon'], 1, DELTA)
                        rdp_demand = compute_rdp_epsilons_gaussian(sigma, ALPHAS)
                    else:
                        rdp_demand = None
                    t_id = next(task_id)

                    init_one_task(
                        task_id=t_id,
                        start_block_idx=t['start_block_index'],
                        end_block_idx=t['end_block_index'],
                        epsilon=t['epsilon'],
                        delta=DELTA,
                        e_rdp=rdp_demand,
                        completion_time=t['completion_time'],
                        cpu_demand=t['cpu_demand'],
                        gpu_demand=t['gpu_demand'],
                        memory_demand=t['memory_demand'],
                    )

    def task(
            self,
            task_id,
            start_block_idx,
            end_block_idx,
            epsilon,
            delta,
            e_rdp,
            completion_time,
            cpu_demand,
            gpu_demand,
            memory_demand,
    ):
        num_blocks_demand = end_block_idx - start_block_idx + 1
        if self.env.config['workload_test.enabled']:
            self.debug(
                task_id,
                'DP demand epsilon=%.2f for blocks No. %s '
                % (epsilon, list(range(start_block_idx, end_block_idx + 1))),
            )
        else:
            self.debug(
                task_id,
                'DP demand epsilon=%.2f for blocks No. %s '
                % (epsilon, range(start_block_idx, end_block_idx).__repr__()),
            )
        self.task_unpublished_count.put(1)
        self.task_ungranted_count.put(1)
        self.task_sleeping_count.put(1)

        t0 = self.env.now

        resource_allocated_event = self.env.event()
        dp_committed_event = self.env.event()
        task_init_event = self.env.event()

        def run_task(task_id, resource_allocated_event):

            assert not resource_allocated_event.triggered
            try:

                yield resource_allocated_event
                resource_alloc_time = self.resource_master.task_state[task_id][
                    "resource_allocate_timestamp"
                ]
                assert resource_alloc_time is not None
                resrc_allocation_wait_duration = resource_alloc_time - t0
                self.debug(
                    task_id,
                    'INFO: Compute Resources allocated after',
                    timedelta(seconds=resrc_allocation_wait_duration),
                )

                self.task_sleeping_count.get(1)
                self.task_running_count.put(1)
            except ResourceAllocFail as err:
                task_abort_timestamp = self.resource_master.task_state[task_id][
                    "task_completion_timestamp"
                ] = -self.env.now
                # note negative sign here
                task_preempted_duration = -task_abort_timestamp - t0
                self.debug(
                    task_id,
                    'WARNING: Resource Allocation fail after',
                    timedelta(seconds=task_preempted_duration),
                )
                self.task_sleeping_count.get(1)
                self.task_abort_count.put(1)
                return 1
            core_running_task = self.env.timeout(
                resource_request_msg["completion_time"]
            )

            def post_completion_callback(event):
                task_completion_timestamp = self.resource_master.task_state[task_id][
                    "task_completion_timestamp"
                ] = self.env.now
                task_completion_duration = task_completion_timestamp - t0
                self.debug(
                    task_id,
                    'Task completed after',
                    timedelta(seconds=task_completion_duration),
                )
                self.task_running_count.get(1)
                self.task_completed_count.put(1)

            try:
                # running task
                yield core_running_task
                post_completion_callback(core_running_task)

                return 0

            except simpy.Interrupt as err:
                assert err.args[0] == ResourceHandlerMessageType.RESRC_HANDLER_INTERRUPT_MSG
                # triggered but not porocessed
                if core_running_task.triggered:
                    # same as post completion_event handling
                    assert not core_running_task.processed
                    post_completion_callback(core_running_task)

                # fixme coverage
                else:
                    task_abort_timestamp = self.resource_master.task_state[task_id][
                        "task_completion_timestamp"
                    ] = -self.env.now
                    # note negative sign here
                    task_preempted_duration = -task_abort_timestamp - t0
                    self.debug(
                        task_id,
                        'Task preempted while running after',
                        timedelta(seconds=task_preempted_duration),
                    )

                    self.task_running_count.get(1)
                    self.task_abort_count.put(1)
                return 1

        def wait_for_dp(task_id, dp_committed_event, epsilon_demand, num_blocks_demand):

            assert not dp_committed_event.triggered
            t0 = self.env.now
            try:
                yield dp_committed_event
                dp_committed_time = self.resource_master.task_state[task_id][
                    "dp_commit_timestamp"
                ] = self.env.now
                dp_committed_duration = dp_committed_time - t0
                self.debug(
                    task_id,
                    'INFO: DP committed after',
                    timedelta(seconds=dp_committed_duration),
                )

                self.task_ungranted_count.get(1)
                self.task_granted_count.put(1)
                task_class = self.resource_master._is_mice_task_dp_demand(
                    epsilon_demand, num_blocks_demand
                )
                if task_class == (True, True):
                    self.tasks_granted_count_s_dp_s_blk.put(1)
                elif task_class == (True, False):
                    self.tasks_granted_count_s_dp_l_blk.put(1)
                elif task_class == (False, True):
                    self.tasks_granted_count_l_dp_s_blk.put(1)
                else:
                    assert task_class == (False, False)
                    self.tasks_granted_count_l_dp_l_blk.put(1)
                return 0

            except (
                    DprequestTimeoutError,
                    InsufficientDpException,
                    StopReleaseDpError,
                    DpBlockRetiredError,
            ) as err:
                assert not dp_committed_event.ok
                dp_rejected_timestamp = self.resource_master.task_state[task_id][
                    "dp_commit_timestamp"
                ] = -self.env.now
                allocation_rej_duration = -dp_rejected_timestamp - t0

                self.debug(
                    task_id,
                    'WARNING: DP commit fails after',
                    timedelta(seconds=allocation_rej_duration),
                    err.__repr__(),
                )
                self.task_dp_rejected_count.put(1)
                return 1

        # listen, wait for allocation
        running_task = self.env.process(run_task(task_id, resource_allocated_event))
        waiting_for_dp = self.env.process(
            wait_for_dp(task_id, dp_committed_event, epsilon, num_blocks_demand)
        )

        # prep allocation request,
        resource_request_msg = {
            "message_type": DpHandlerMessageType.ALLOCATION_REQUEST,
            "task_id": task_id,
            "cpu": cpu_demand,
            "memory": memory_demand,
            "gpu": gpu_demand,
            "epsilon": epsilon,
            "delta": delta,
            # todo maybe exclude EOL blocks?
            "e_rdp": e_rdp,  # a list of rdp demand
            "block_idx": list(
                range(start_block_idx, end_block_idx + 1)
            ),  # choose latest num_blocks_demand
            "completion_time": completion_time,
            "resource_allocated_event": resource_allocated_event,
            "dp_committed_event": dp_committed_event,
            'task_init_event': task_init_event,
            "user_id": None,
            "model_id": None,
            'execution_proc': running_task,
            'waiting_for_dp_proc': waiting_for_dp,
        }
        # send allocation request, note, do it when child process is already listening
        self.resource_master.mail_box.put(resource_request_msg)

        t0 = self.env.now
        if self.db:
            assert isinstance(task_id, int)
            assert isinstance(resource_request_msg["block_idx"][0], int)
            assert isinstance(num_blocks_demand, int)
            assert isinstance(resource_request_msg["epsilon"], float)
            assert isinstance(resource_request_msg["cpu"], int)
            assert isinstance(resource_request_msg["gpu"], (int, NoneType))
            assert isinstance(resource_request_msg["memory"], (int, NoneType))
            assert isinstance(t0, (float, int))

            def db_init_task():
                self.db.execute(
                    'INSERT INTO tasks '
                    '(task_id, start_block_id, num_blocks, epsilon, cpu, gpu, memory, '
                    'start_timestamp) '
                    'VALUES (?,?,?,?,?,?,?,?)',
                    (
                        task_id,
                        resource_request_msg["block_idx"][0],
                        num_blocks_demand,
                        resource_request_msg["epsilon"],
                        resource_request_msg["cpu"],
                        resource_request_msg["gpu"],
                        resource_request_msg["memory"],
                        t0,
                    ),
                )

            for i in range(20):  # try 20 times
                try:
                    db_init_task()
                    break
                except Exception as e:
                    time.sleep(0.3)
            else:
                raise (e)

        dp_grant, task_exec = yield self.env.all_of([waiting_for_dp, running_task])

        if dp_grant.value == 0:
            # verify, if dp granted, then task must be completed.
            assert task_exec.value == 0
            self.resource_master.task_state[task_id][
                "task_publish_timestamp"
            ] = self.env.now
            self.task_unpublished_count.get(1)
            self.task_published_count.put(1)
            publish_duration = self.env.now - t0
            self.debug(
                task_id,
                "INFO: task get published after ",
                timedelta(seconds=publish_duration),
            )
        else:
            assert dp_grant.value == 1
            publish_fail_duration = self.env.now - t0
            self.debug(
                task_id,
                "WARNING: task fail to publish after ",
                timedelta(seconds=publish_fail_duration),
            )
            self.resource_master.task_state[task_id]["task_publish_timestamp"] = None

        if self.db:
            # verify iff cp commit fail <=> no publish
            if (
                    self.resource_master.task_state[task_id]["task_publish_timestamp"]
                    is None
            ):
                assert (
                        self.resource_master.task_state[task_id]["dp_commit_timestamp"] <= 0
                )

            if self.resource_master.task_state[task_id]["dp_commit_timestamp"] < 0:
                assert (
                        self.resource_master.task_state[task_id]["task_publish_timestamp"]
                        is None
                )

            assert isinstance(
                self.resource_master.task_state[task_id]["dp_commit_timestamp"],
                (float, int),
            )
            assert isinstance(
                self.resource_master.task_state[task_id]["resource_allocate_timestamp"],
                (float, NoneType, int),
            )
            assert isinstance(
                self.resource_master.task_state[task_id]["task_completion_timestamp"],
                (float, int),
            )
            assert isinstance(
                self.resource_master.task_state[task_id]["task_publish_timestamp"],
                (float, NoneType, int),
            )

            def db_update_task():
                self.db.execute(
                    'UPDATE tasks '
                    'set dp_commit_timestamp = ?, resource_allocation_timestamp = ?, completion_timestamp = ?, publish_timestamp = ?'
                    'where task_id= ?',
                    (
                        self.resource_master.task_state[task_id]["dp_commit_timestamp"],
                        self.resource_master.task_state[task_id][
                            "resource_allocate_timestamp"
                        ],
                        self.resource_master.task_state[task_id][
                            "task_completion_timestamp"
                        ],
                        self.resource_master.task_state[task_id][
                            "task_publish_timestamp"
                        ],
                        task_id,
                    ),
                )

            for i in range(20):  # try 20 times
                try:
                    db_update_task()
                    break
                except Exception as e:
                    time.sleep(0.3)
            else:
                raise (e)
        return

    def get_result_hook(self, result):

        if not self.db:
            return
        result['succeed_tasks_total'] = self.db.execute(
            'SELECT COUNT() FROM tasks  WHERE  dp_commit_timestamp >=0'
        ).fetchone()[0]
        result['succeed_tasks_s_dp_s_blk'] = self.db.execute(
            'SELECT COUNT() FROM tasks  WHERE  dp_commit_timestamp >=0 and epsilon < ? and num_blocks < ?',
            (self.resource_master._avg_epsilon, self.resource_master._avg_num_blocks),
        ).fetchone()[0]

        result['succeed_tasks_s_dp_l_blk'] = self.db.execute(
            'SELECT COUNT() FROM tasks  WHERE  dp_commit_timestamp >=0 and epsilon < ? and num_blocks > ?',
            (self.resource_master._avg_epsilon, self.resource_master._avg_num_blocks),
        ).fetchone()[0]

        result['succeed_tasks_l_dp_s_blk'] = self.db.execute(
            'SELECT COUNT() FROM tasks  WHERE  dp_commit_timestamp >=0 and epsilon > ? and num_blocks < ?',
            (self.resource_master._avg_epsilon, self.resource_master._avg_num_blocks),
        ).fetchone()[0]

        result['succeed_tasks_l_dp_l_blk'] = self.db.execute(
            'SELECT COUNT() FROM tasks  WHERE  dp_commit_timestamp >=0 and epsilon > ? and num_blocks > ?',
            (self.resource_master._avg_epsilon, self.resource_master._avg_num_blocks),
        ).fetchone()[0]

        result['succeed_tasks_per_hour'] = result['succeed_tasks_total'] / (
                self.env.time() / 3600
        )

        # not exact cal for median
        sql_duration_percentile = """
        with nt_table as
         (
             select (%s - start_timestamp) AS dp_allocation_duration, ntile(%d) over (order by (dp_commit_timestamp - start_timestamp)  desc) ntile
             from tasks
             WHERE dp_commit_timestamp >=0
         )

        select avg(a)from (
                 select min(dp_allocation_duration) a
                 from nt_table
                 where ntile = 1
        
                 union
                 select max(dp_allocation_duration) a
                 from nt_table
                 where ntile = 2
     )"""
        result['dp_allocation_duration_avg'] = self.db.execute(
            'SELECT AVG(dp_commit_timestamp - start_timestamp) as dur FROM tasks WHERE dp_commit_timestamp >=0 '
        ).fetchone()[0]

        result['dp_allocation_duration_min'] = self.db.execute(
            'SELECT MIN(dp_commit_timestamp - start_timestamp) as dur FROM tasks WHERE dp_commit_timestamp >=0'
        ).fetchone()[0]

        result['dp_allocation_duration_max'] = self.db.execute(
            'SELECT MAX(dp_commit_timestamp - start_timestamp) as dur FROM tasks WHERE  dp_commit_timestamp >=0'
        ).fetchone()[0]


        result['dp_allocation_duration_Median'] = (
            self.db.execute(
                sql_duration_percentile % ('dp_commit_timestamp', 2)
            ).fetchone()[0]
            if result['succeed_tasks_total'] >= 2
            else None
        )

        result['dp_allocation_duration_P99'] = (
            self.db.execute(
                sql_duration_percentile % ('dp_commit_timestamp', 100)
            ).fetchone()[0]
            if result['succeed_tasks_total'] >= 100
            else None
        )

        result['dp_allocation_duration_P999'] = (
            self.db.execute(
                sql_duration_percentile % ('dp_commit_timestamp', 1000)
            ).fetchone()[0]
            if result['succeed_tasks_total'] >= 1000
            else None
        )

        result['dp_allocation_duration_P9999'] = (
            self.db.execute(
                sql_duration_percentile % ('dp_commit_timestamp', 10000)
            ).fetchone()[0]
            if result['succeed_tasks_total'] >= 10000
            else None
        )

if __name__ == '__main__':
    pass
