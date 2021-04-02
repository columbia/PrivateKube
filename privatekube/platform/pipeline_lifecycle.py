import json

from privacy_resource_client import PIPELINE_NAMESPACE, PrivacyResourceClient


class PipelineInfo:
    def __init__(self, run_info):
        self.run_info = run_info
        workflow_manifest = run_info.pipeline_runtime.workflow_manifest
        self.workflow = json.loads(workflow_manifest)

    def get_workflow_name(self):
        return self.workflow["metadata"]["name"]

    @staticmethod
    def create(run_info):
        info = PipelineInfo(run_info)
        if "name" not in info.workflow["metadata"]:
            return None
        return info


class WorkflowStatistic:
    def __init__(self, client: PrivacyResourceClient):
        self.client = client
        self.kfp_client = client.kfp_client

    def get_workflow(self, namespace, name):
        workflow = self.client.get_workflow(namespace, name)
        return workflow

    def get_pipeline_info(self, run_id):
        run_info = self.kfp_client.get_run(run_id)
        return PipelineInfo.create(run_info)


def extract_start_end_time_of_workflow(workflow):
    return workflow["status"]["startedAt"], workflow["status"]["finishedAt"]


if __name__ == "__main__":
    client = PrivacyResourceClient()
    workflow_stat = WorkflowStatistic(client)
    workflow = workflow_stat.get_workflow(PIPELINE_NAMESPACE, "test-taxi-claims-nvkq6")
    print(workflow)
    start_time, end_time = extract_start_end_time_of_workflow(workflow)
    print(f"Started at: {start_time}. Finished at: {end_time}")
