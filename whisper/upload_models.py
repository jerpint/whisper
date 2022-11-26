"""Upload models to the huggingface hub.

The goal of this program is to upload pre-trained whisper models directly to the huggingface hub.
First, a new huggingface model hub is created if it doesn't exist.
A model is then downloaded from the original OpenAI repo and uploaded to the hub afterwards.

Typical usage example:

    python upload_models.py
"""

import os
import whisper

from huggingface_hub import create_repo, HfApi, CommitOperationAdd, Repository


def create_repository(repo_id: str) -> str:
    """Creates a huggingface model repository if it doesn't already exist.

    You should be logged in to huggingface-cli. See:
    https://huggingface.co/docs/huggingface_hub/how-to-manage

    Args:
        repo_id (str): the name of the huggingface repository you want to create,
            e.g. <USER>/whisper

    Returns:
        str: the full repo_url on huggingface
    """
    print(f"Initializing {repo_id}...")
    repo_url = create_repo(repo_id, repo_type="model", exist_ok=True)
    return repo_url


def download_pretrained_model(model_name: str, output_dir: str) -> str:
    """Download the pre-trained model using openai's API.

    Args:
        model_name (str): should be a valid model available from open-ai.
            see here for full list of available models: https://github.com/jerpint/whisper/blob/main/whisper/__init__.py#L17
        output_dir (str): local path to save the model to.

    Returns:
        str: the path of the downloaded model.
    """
    # Check that the model name is valid
    assert (
        model_name in whisper._MODELS.keys()
    ), f"Model can be one of f{whisper._MODELS.keys()}"

    # create folder if it doesn't exist
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # download the model
    url = whisper._MODELS[model_name]
    print(f"Downloading model '{model_name}.pt' from {url}")
    whisper._download(url, root=output_dir, in_memory=False)

    return os.path.join(output_dir, model_name + ".pt")


def upload_model_to_hub(local_path: str, hub_path: str, repo_id):
    """Upload the model to the huggingface hub.

    Args:
        local_path (str): path to model file to upload
        hub_path (str): path that will appear on hub
        repo_id (_type_): repo to upload to (e.g. "<USER>/whisper")
    """

    api = HfApi()

    print(f"Uploading {local_path} to {hub_path} on {repo_id}...")

    operations = [
        CommitOperationAdd(path_in_repo=hub_path, path_or_fileobj=local_path),
    ]

    api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message=f"Upload model {hub_path}",
    )


if __name__ == "__main__":
    repo_id = "jerpint/whisper"
    local_dir = "models/"
    model_name = "small"
    hub_path = model_name + ".pt"

    repo_url = create_repository(repo_id=repo_id)
    model_path = download_pretrained_model(model_name=model_name, output_dir=local_dir)
    upload_model_to_hub(local_path=model_path, hub_path=hub_path, repo_id=repo_id)
