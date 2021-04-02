from copy import deepcopy
from uuid import uuid4
from loguru import logger
from kubernetes.client.rest import ApiException

from stoppable_thread import StoppableThread
from privacy_resource_client import (
    PrivacyResourceClient,
    PRIVACY_BUDGET_CLAIM_GROUP,
    PRIVACY_BUDGET_CLAIM_VERSION,
)
from privacy_budget import (
    PrivacyBudget,
    RenyiBudget,
    EpsDelBudget,
)
from timer import Timer

LOWER_LIMIT = 1e-12


def get_acquired_budgets(claim):
    if "status" not in claim or "acquiredBudgets" not in claim["status"]:
        return None
    return claim["status"]["acquiredBudgets"]


def get_comitted_budgets(claim):
    if "status" not in claim or "committedBudgets" not in claim["status"]:
        return None
    return claim["status"]["committedBudgets"]


class PrivacyClaim(object):
    def __init__(
        self,
        client: PrivacyResourceClient,
        timer: Timer,
        dataset,
        block_start_time,
        block_end_time,
        min_number_of_blocks,
        max_number_of_blocks,
        budget_wait_time,
        expected_budget: PrivacyBudget,
        namespace="privacy-example",
        claim_name=None,
        identifier=None,
    ):
        """
        :param client: privacy resource client
        :param timer: timer used in the evaluation platform
        :param dataset:
        :param block_start_time:
        :param block_end_time:
        :param min_number_of_blocks:
        :param budget_wait_time:
        :param expected_epsilon:
        :param expected_delta:
        """
        self.client = client
        self.timer = timer
        self.dataset = dataset
        self.block_start_time = block_start_time
        self.block_end_time = block_end_time
        self.min_number_of_blocks = min_number_of_blocks
        self.max_number_of_blocks = max_number_of_blocks
        self.budget_wait_time = budget_wait_time
        self.expected_privacy_budget = expected_budget
        self.namespace = namespace
        self.identifier = identifier
        if claim_name is None:
            self._claim_name = str(uuid4())  # randomized
        else:
            self._claim_name = claim_name
        self.results = {}

    @property
    def expected_budget(self):
        # return {
        #     "epsilon": self.expected_epsilon,
        #     "delta": self.expected_delta,
        # }
        return self.expected_privacy_budget.to_dict()

    @property
    def name(self):
        return self._claim_name

    def create_claim(self):
        spec = {
            "requests": [
                {
                    "allocateRequest": {
                        "dataset": self.dataset,
                        "conditions": [
                            {
                                "attribute": "startTime",
                                "numericValue": str(self.block_start_time),
                                "operation": ">=",
                            },
                            {
                                "attribute": "endTime",
                                "numericValue": str(self.block_end_time),
                                "operation": "<=",
                            },
                        ],
                        "minNumberOfBlocks": self.min_number_of_blocks,
                        "maxNumberOfBlocks": self.max_number_of_blocks,
                        "expectedBudget": {"constant": self.expected_budget},
                        "timeout": self.budget_wait_time * 1000,
                    }
                }
            ]
        }

        return {
            "apiVersion": PRIVACY_BUDGET_CLAIM_GROUP
            + "/"
            + PRIVACY_BUDGET_CLAIM_VERSION,
            "kind": "PrivacyBudgetClaim",
            "metadata": {"name": self._claim_name, "namespace": self.namespace},
            "spec": spec,
        }

    def create_and_upload_claim(self):
        while True:
            claim = self.create_claim()
            try:
                return self.client.create_privacy_budget_claim(self.namespace, claim)
            except ApiException as e:
                if e.reason == "Conflict":
                    continue

    def run(self):
        logger.info(f"{self._claim_name} is ready to run")
        self.create_and_upload_claim()
        start_time = self.timer.current()
        # wait for budget
        while True:
            try:
                claim = self.client.get_privacy_budget_claim(
                    self.namespace, self._claim_name
                )
            except ApiException:
                continue
            if (
                "status" in claim
                and "responses" in claim["status"]
                and len(claim["status"]["responses"]) == 1
                and claim["status"]["responses"][0]["state"] != "pending"
            ):
                break
            self.timer.sleep(1)

        finish_time = self.timer.current()
        self.results["actual_start_time"] = start_time
        self.results["actual_finish_time"] = finish_time
        self.results["actual_wait_time"] = finish_time - start_time

        allocate_response = claim["status"]["responses"][0]["allocateResponse"]

        self.results["finish_time"] = allocate_response["finishTime"]
        if "startTime" in allocate_response:
            self.results["start_time"] = allocate_response["startTime"]
        else:
            self.results["start_time"] = allocate_response["finishTime"]
        self.results["wait_time"] = (
            self.results["finish_time"] - self.results["start_time"]
        )

        self.results["state"] = claim["status"]["responses"][0]["state"]
        if "error" in claim["status"]["responses"][0]:
            self.results["error"] = claim["status"]["responses"][0]["error"]
        else:
            self.results["error"] = None

        self.results["acquired_budgets"] = get_acquired_budgets(claim)

        logger.info(f"{self._claim_name} is finished. Result: {self.results}")
        return 0

    def delete_claim(self):
        self.client.delete_privacy_budget_claim(self.namespace, self.name)


class PrivacyClaimThread(StoppableThread):
    def __init__(self, model: PrivacyClaim):
        super().__init__(target=model.run, args=tuple())
        self.model = model


if __name__ == "__main__":
    client = PrivacyResourceClient()
    timer = Timer()

    model = PrivacyClaim(
        client, timer, "taxi", 0, 1000, 1, 10, 10, EpsDelBudget(0.2, 1e-4)
    )
    model.run()
