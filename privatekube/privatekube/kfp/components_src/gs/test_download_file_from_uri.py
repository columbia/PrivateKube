import kfp
from kfp import dsl
from kfp import gcp

# Add your own configuration here
ARTIFACT_PATH = ""
GCS_PATH = ""

if __name__ == "__main__":

    op = kfp.components.load_component_from_file(
        ARTIFACT_PATH + "/download_file_from_uri.yaml"
    )

    @dsl.pipeline(name="Test component.", description="")
    def pipeline(
        input_path,
    ):
        _op = op(
            input_path=input_path,
        ).apply(gcp.use_gcp_secret("user-gcp-sa"))

    # Specify pipeline argument values
    arguments = {"input_path": GCS_PATH}

    # Submit a pipeline run
    kfp.Client().create_run_from_pipeline_func(pipeline, arguments=arguments)
