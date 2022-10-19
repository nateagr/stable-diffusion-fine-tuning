import os

import fsspec
from diffusers import StableDiffusionPipeline


def load_stable_diffusion_pipeline(pipeline_hdfs_path: str, download_dir = "."):
    pipeline_local_path = os.path.join(download_dir, os.path.basename(pipeline_hdfs_path))
    fs, _ = fsspec.core.url_to_fs(pipeline_hdfs_path, use_listings_cache=False)
    fs.get(pipeline_hdfs_path, pipeline_local_path, recursive=True)
    return StableDiffusionPipeline.from_pretrained(
        pipeline_local_path,
        use_auth_token=True
    )
