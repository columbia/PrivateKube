import kfp
from kfp import dsl
from kfp import gcp

# Add your own configuration here
ARTIFACT_PATH = ""
GCS_PATH = ""

if __name__ == "__main__":

    download_op = kfp.components.load_component_from_file(
        ARTIFACT_PATH + "/download_dir_from_uri.yaml"
    )
    list_op = kfp.components.load_component_from_file(ARTIFACT_PATH + "/list_dir.yaml")

    @dsl.pipeline(name="Test component.", description="")
    def pipeline(
        input_dir,
    ):
        download_task = download_op(
            input_dir=input_dir,
        ).apply(gcp.use_gcp_secret("user-gcp-sa"))

        list_task = list_op(input_dir=download_task.outputs["output_dir"])

    # Specify pipeline argument values
    arguments = {"input_dir": GCS_PATH}

    # Submit a pipeline run
    kfp.Client().create_run_from_pipeline_func(pipeline, arguments=arguments)
