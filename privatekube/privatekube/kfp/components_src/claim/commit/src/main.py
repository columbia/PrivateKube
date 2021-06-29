#!/usr/bin/env python

import argparse
import os
import sys
import time
import uuid
import gcsfs
import json
import yaml
from pathlib import Path

from kubernetes import client
from kubernetes import config as kconfig

# Linear timeout increment (in seconds) between two checks
TIMEOUT_INCREMENT = 0.5


def create_uniform_budgets(claim, budget):
    """
    Input:
    - a claim dict that contains a list of blocks and individual budgets
    - a global budget on a merged dataset
    Output:
    - a dictionary of budgets to consume for each block
    """
    print(claim)
    epsilon, delta = budget["epsilon"], budget["delta"]

    # Browse all the responses to gather the budgets, and commit them all
    acquired_budgets = {}
    for response in claim["status"]["responses"]:
        if "allocateResponse" in response:
            acquired_budgets.update(response["allocateResponse"]["budgets"])
    print(acquired_budgets)

    consumed_budgets = {}
    for i, budget in enumerate(acquired_budgets.items()):
        print("budget")
        print(budget)

        block = budget[0]
        d = budget[1]["acquired"]["epsDel"]["delta"]
        e = budget[1]["acquired"]["epsDel"]["epsilon"]

        if i == 0:
            acquired_delta = d
            acquired_epsilon = e
        else:
            assert e == acquired_epsilon and d == acquired_delta

        consumed_uniform_budget = {"epsDel": {"epsilon": epsilon, "delta": delta}}

        consumed_budgets[block] = consumed_uniform_budget

    return consumed_budgets


def create_consume_request(policy, budget):

    request_id = str(uuid.uuid4().hex)

    request_dict = {
        "identifier": request_id,
        "consumeRequest": {"policy": policy, "consume": budget},
    }

    return request_dict


def main(args):

    # Connect to the cluster
    os.system(
        f"gcloud container clusters get-credentials {args.cluster_name} --zone={args.zone}"
    )
    kconfig.load_kube_config()
    api = client.CustomObjectsApi()

    # Retrieve the privacy claim. Kubeflow opens the JSON files for us.
    claim = json.loads(args.claim)
    budget = json.loads(args.budget)

    # Create a new commit request and submit a patch
    budgets_dict = create_uniform_budgets(claim, budget)
    request_dict = create_consume_request(
        policy="allOrNothing=false", budget=budgets_dict
    )

    request_id = request_dict["identifier"]
    namespace = claim["metadata"]["namespace"]
    claim_name = claim["metadata"]["name"]
    patch_dict = [{"op": "add", "path": "/spec/requests/-", "value": request_dict}]

    print(patch_dict)
    print(json.dumps(patch_dict))

    os.system(
        f"kubectl patch pbc {claim_name} -n {namespace} --type json --patch '{json.dumps(patch_dict)}'"
    )
    print(f"Claim {claim_name} patched.")

    # Watch the request until there is a response
    start_time = time.time()
    success = False
    index = None
    while time.time() < start_time + args.timeout and not success:

        get_claim = api.get_namespaced_custom_object(
            group="columbia.github.com",
            version="v1",
            name=claim_name,
            namespace=namespace,
            plural="privacybudgetclaims",
        )

        try:
            # Find the request
            if index is None:
                responses = get_claim["status"]["responses"]
                print(responses)
                for i, response in enumerate(responses):
                    if response["identifier"] == request_id:
                        index = i
                # If the index is not there yet, the loop will fail and try again after some sleep

            state = get_claim["status"]["responses"][index]["state"]
            print(state)
            if state == "success":
                print(f"Request {request_id} succeeded.")
                success = True
            elif state == "failure":
                raise Exception(
                    "Response error: the commit request failed.\n" + str(get_claim)
                )

        except Exception as e:
            print("Waiting for response.")
            print(e)
            time.sleep(TIMEOUT_INCREMENT)

    # On timeout, create a component failure
    if not success:
        raise Exception("Timeout error: the request was not answered on time.")


if __name__ == "__main__":

    # Documentation for args: see .yaml component
    parser = argparse.ArgumentParser()

    # Input Values
    parser.add_argument("--claim", type=str, required=True)
    parser.add_argument("--budget", type=str, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--cluster_name", type=str, required=True)
    parser.add_argument("--zone", type=str, required=True)

    args = parser.parse_args()
    main(args)
