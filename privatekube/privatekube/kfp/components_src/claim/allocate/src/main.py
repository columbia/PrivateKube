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


def create_claim(namespace, allocation_request, claim_name=None):
    if claim_name is None:
        claim_name = "claim-" + str(uuid.uuid4().hex)

    claim_dict = {
        "apiVersion": "columbia.github.com/v1",
        "kind": "PrivacyBudgetClaim",
        "metadata": {"name": claim_name, "namespace": namespace},
        "spec": {"requests": [allocation_request]},
    }
    return claim_dict


def create_timespan_allocation_request(
    dataset, start_time, end_time, epsilon, delta, n_blocks: int
):

    request_id = str(uuid.uuid4().hex)

    budget = {"epsDel": {"epsilon": epsilon, "delta": delta}}

    request_dict = {
        "identifier": request_id,
        "allocateRequest": {
            "dataset": dataset,
            "conditions": [
                {
                    "attribute": "startTime",
                    "numericValue": start_time,
                    "operation": ">=",
                },
                {
                    "attribute": "endTime",
                    "numericValue": end_time,
                    "operation": "<",
                },
            ],
            "minNumberOfBlocks": n_blocks,
            "maxNumberOfBlocks": n_blocks,
            "expectedBudget": {"constant": budget},
        },
    }

    return request_dict


def get_datasource(api, block_reference):
    """
    Ex:
    input: privacy-example/block-2015-01-01
    output:  gs://private-systems-1887eadb-public/processed/days/2015-01-01.csv
    """
    namespace, name = block_reference.split("/")
    get_block = api.get_namespaced_custom_object(
        group="columbia.github.com",
        version="v1",
        name=name,
        namespace=namespace,
        plural="privatedatablocks",
    )
    return get_block["spec"]["dataSource"]


def main(args):

    with open(Path(__file__).parent.joinpath("cluster_secrets.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # Prepare Kubernetes
    # os.system(
    #     "gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS"
    # )
    os.system(
        f"gcloud container clusters get-credentials {config['cluster_name']} --zone={config['zone']}"
    )
    kconfig.load_kube_config()
    api = client.CustomObjectsApi()

    # Create a new claim and submit an allocation request
    request_dict = create_timespan_allocation_request(
        args.dataset,
        args.start_time,
        args.end_time,
        args.epsilon,
        args.delta,
        args.n_blocks,
    )

    request_id = request_dict["identifier"]

    claim_dict = create_claim(args.namespace, request_dict)

    claim_name = claim_dict["metadata"]["name"]

    api.create_namespaced_custom_object(
        group="columbia.github.com",
        version="v1",
        namespace=args.namespace,
        plural="privacybudgetclaims",
        body=claim_dict,
    )
    print(f"Claim {claim_name} created.")

    # Watch the allocation request until there is a response

    start_time = time.time()
    success = False
    index = None
    while time.time() < start_time + args.timeout and not success:

        get_claim = api.get_namespaced_custom_object(
            group="columbia.github.com",
            version="v1",
            name=claim_name,
            namespace=args.namespace,
            plural="privacybudgetclaims",
        )

        try:
            print(get_claim)
            # index = int(get_claim["metadata"]["annotations"]["request_" + request_id])

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
                    "Response error: the allocation request failed.\n" + str(get_claim)
                )

        except Exception as e:
            print("Waiting for response.")
            print(e)
            time.sleep(TIMEOUT_INCREMENT)

    # On timeout, create a component failure
    if not success:
        raise Exception("Timeout error: the claim was not allocated on time.")

    # Local output files
    for output_path in [args.output_data, args.output_budget, args.output_claim]:
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

    # Get the datasources of the blocks and write them in a file (one URI per line)
    with open(args.output_data, "w") as f:
        acquired_budgets = get_claim["status"]["acquiredBudgets"]
        for block_reference, budget in acquired_budgets.items():
            assert (
                budget["epsDel"]["epsilon"] == args.epsilon
                and budget["epsDel"]["delta"] == args.delta
            )

            datasource = get_datasource(api, block_reference)
            print(f"Datasource: {datasource}")
            f.write(datasource)
            f.write("\n")

    # Write the rest of the outputs locally too
    with open(args.output_budget, "w") as output_json:
        budget = {"epsilon": args.epsilon, "delta": args.delta}
        json.dump(budget, output_json)

    with open(args.output_claim, "w") as output_json:
        json.dump(get_claim, output_json)


if __name__ == "__main__":

    # Documentation for args: see .yaml component
    parser = argparse.ArgumentParser()

    # Input Values
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--namespace", type=str, required=True)
    parser.add_argument("--start_time", type=str, required=True)
    parser.add_argument("--end_time", type=str, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--n_blocks", type=int, required=True)
    parser.add_argument("--timeout", type=int, required=True)

    # Output Paths (provided by Kubeflow to write artifacts)
    parser.add_argument("--output_data", type=str, required=True)
    parser.add_argument("--output_budget", type=str, required=True)
    parser.add_argument("--output_claim", type=str, required=True)

    args = parser.parse_args()
    main(args)
