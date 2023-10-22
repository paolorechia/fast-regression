# Option 2: Using Ray Jobs API (Python SDK)
from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient("http://0.0.0.0:8265")

runtime_env = {
    "pip": [
        "scikit-learn",
        "skl2onnx",
        "onnxruntime",
        "pyspark>=3.5",
        "pandas>=2",
        "pyarrow>=13.0.0",
        "ray[data,train,tune,serve]",
        "wheel",
        "setuptools_scm",
        "ray[default]",
        "ray[client]",
    ],
    "working_dir": "./",
}
job_id = client.submit_job(
    entrypoint="python predict_remote_ray.py",
    runtime_env=runtime_env,
)
