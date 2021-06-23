import os
import sys

from kubernetes.client import CustomObjectsApi
from kubernetes.config import load_kube_config

# set the CLOUD SDK to use the current python interpreter. This is important when you use a virtual env.
os.environ["CLOUDSDK_PYTHON"] = sys.executable

load_kube_config(config_file="~/.kube/config")
print("Successfully read the kube config.")

PIPELINE_GROUP = "argoproj.io"
PIPELINE_VERSION = "v1alpha1"
PIPELINE_PLURAL = "workflows"
PIPELINE_NAMESPACE = "kubeflow"

DATA_BLOCK_GROUP = "columbia.github.com"
DATA_BLOCK_VERSION = "v1"
DATA_BLOCK_PLURAL = "privatedatablocks"

PRIVACY_BUDGET_CLAIM_GROUP = "columbia.github.com"
PRIVACY_BUDGET_CLAIM_VERSION = "v1"
PRIVACY_BUDGET_CLAIM_PLURAL = "privacybudgetclaims"


class PrivacyResourceClient(object):
    def __init__(self):
        self.api = CustomObjectsApi()

        # self.kfp_client = kfp.Client()

    def create_data_block(self, namespace, block):
        return self.api.create_namespaced_custom_object(
            group=DATA_BLOCK_GROUP,
            version=DATA_BLOCK_VERSION,
            namespace=namespace,
            plural=DATA_BLOCK_PLURAL,
            body=block,
        )

    def update_data_block_status(self, namespace, block):
        return self.api.replace_namespaced_custom_object_status(
            group=DATA_BLOCK_GROUP,
            version=DATA_BLOCK_VERSION,
            namespace=namespace,
            plural=DATA_BLOCK_PLURAL,
            name=block["metadata"]["name"],
            body=block,
        )

    def delete_data_block(self, namespace, name):
        return self.api.delete_namespaced_custom_object(
            group=DATA_BLOCK_GROUP,
            version=DATA_BLOCK_VERSION,
            namespace=namespace,
            plural=DATA_BLOCK_PLURAL,
            name=name,
        )

    def get_data_block(self, namespace, name):
        return self.api.get_namespaced_custom_object(
            group=DATA_BLOCK_GROUP,
            version=DATA_BLOCK_VERSION,
            namespace=namespace,
            plural=DATA_BLOCK_PLURAL,
            name=name,
        )

    def list_data_blocks(self, namespace):
        blocks = self.api.list_namespaced_custom_object(
            group=DATA_BLOCK_GROUP,
            version=DATA_BLOCK_VERSION,
            namespace=namespace,
            plural=DATA_BLOCK_PLURAL,
        )
        return blocks["items"]

    def create_privacy_budget_claim(self, namespace, claim):
        return self.api.create_namespaced_custom_object(
            group=PRIVACY_BUDGET_CLAIM_GROUP,
            version=PRIVACY_BUDGET_CLAIM_VERSION,
            namespace=namespace,
            plural=PRIVACY_BUDGET_CLAIM_PLURAL,
            body=claim,
        )

    def update_privacy_budget_claim(self, namespace, claim):
        return self.api.replace_namespaced_custom_object(
            group=PRIVACY_BUDGET_CLAIM_GROUP,
            version=PRIVACY_BUDGET_CLAIM_VERSION,
            namespace=namespace,
            plural=PRIVACY_BUDGET_CLAIM_PLURAL,
            name=claim["metadata"]["name"],
            body=claim,
        )

    def delete_privacy_budget_claim(self, namespace, name):
        return self.api.delete_namespaced_custom_object(
            group=PRIVACY_BUDGET_CLAIM_GROUP,
            version=PRIVACY_BUDGET_CLAIM_VERSION,
            namespace=namespace,
            plural=PRIVACY_BUDGET_CLAIM_PLURAL,
            name=name,
        )

    def get_privacy_budget_claim(self, namespace, name):
        return self.api.get_namespaced_custom_object(
            group=PRIVACY_BUDGET_CLAIM_GROUP,
            version=PRIVACY_BUDGET_CLAIM_VERSION,
            namespace=namespace,
            plural=PRIVACY_BUDGET_CLAIM_PLURAL,
            name=name,
        )

    def list_privacy_budget_claims(self, namespace):
        claims = self.api.list_namespaced_custom_object(
            group=PRIVACY_BUDGET_CLAIM_GROUP,
            version=PRIVACY_BUDGET_CLAIM_VERSION,
            namespace=namespace,
            plural=PRIVACY_BUDGET_CLAIM_PLURAL,
        )
        return claims["items"]

    def get_workflow(self, namespace, name):
        workflow = self.api.get_namespaced_custom_object(
            group=PIPELINE_GROUP,
            version=PIPELINE_VERSION,
            namespace=namespace,
            plural=PIPELINE_PLURAL,
            name=name,
        )
        return workflow

    def delete_workflow(self, name):
        self.api.delete_namespaced_custom_object(
            group=PIPELINE_GROUP,
            version=PIPELINE_VERSION,
            namespace=PIPELINE_NAMESPACE,
            plural=PIPELINE_PLURAL,
            name=name,
        )


# def get_kfp_client() -> kfp.Client:
#     return kfp.Client()
