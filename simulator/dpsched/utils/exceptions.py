

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

class InsufficientDpException(RuntimeError):
    # remaining dp in a block is not enough for a task commit
    pass


class RejectDpPermissionError(RuntimeError):
    pass


class StopReleaseDpError(RuntimeError):
    # time based release run out of DP
    pass


class DpBlockRetiredError(RuntimeError):
    pass


class DprequestTimeoutError(RuntimeError):
    pass


class ResourceAllocFail(RuntimeError):
    pass


class TaskPreemptedError(RuntimeError):
    pass


class RejectResourcePermissionError(RuntimeError):
    pass
