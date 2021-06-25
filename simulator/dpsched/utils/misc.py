#! /usr/bin/python

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

def defuse(event):
    def set_defused(evt):
        evt.defused = True

    if not event.processed:
        event.callbacks.append(set_defused)


def max_min_fair_allocation(demand, capacity):

    if sum(demand.values()) <= capacity:
        return demand

    equal_share = capacity / len(demand)
    if min(demand.values()) > equal_share:
        return {u:equal_share for u in demand}

    subdesired = {}
    partial_min_max_share = {}

    for u, d in demand.items():
        if d <= equal_share:
            partial_min_max_share[u] = d
        else:
            subdesired[u] = d
    partial_share_sum = sum(partial_min_max_share.values())
    partial_min_max_share.update(max_min_fair_allocation(subdesired, capacity - partial_share_sum))
    return partial_min_max_share

