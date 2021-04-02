import kfp
from kfp import dsl
from kfp import gcp

# Add your own configuration here
ARTIFACT_PATH = ""
GCS_PATH = ""


if __name__ == "__main__":

    down_op = kfp.components.load_component_from_file(
        ARTIFACT_PATH + "/download_file_from_uri.yaml"
    )

    up_op = kfp.components.load_component_from_file(
        ARTIFACT_PATH + "/upload_file_to_uri.yaml"
    )

    @dsl.pipeline(name="Test component.", description="")
    def pipeline(input_path, output_path):
        down_task = down_op(
            input_path=input_path,
        ).apply(gcp.use_gcp_secret("user-gcp-sa"))

        up_task = up_op(
            input_artifact=down_task.outputs["output_path"], output_path=output_path
        ).apply(gcp.use_gcp_secret("user-gcp-sa"))

    # Specify pipeline argument values
    arguments = {
        "input_path": GCS_PATH,
        "output_path": GCS_PATH + ".copy",
    }

    # Submit a pipeline run
    kfp.Client().create_run_from_pipeline_func(pipeline, arguments=arguments)
