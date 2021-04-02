# Subpipeline

*To be released.*

Input:
- GCS path to a pipeline file (yaml, zip, tar.gz)
- GCS path to a Yaml file containing run arguments
Output:
- A unique (tmp) GCS path to the outputs of the pipeline (assuming it is outputing a Yaml at 'pipeline_output_path').

## Serialized input arguments

Input:
- GCS path to a pipeline file (yaml, zip, tar.gz)
- Parameter: serialized Yaml arguments
Output:
- A unique (tmp) GCS path to the outputs of the pipeline (assuming it is outputing a Yaml at 'pipeline_output_path').