from huggingface_hub import hf_hub_download

REPO_ID = "nickrosh/Evol-Instruct-Code-80k-v1"
FILENAME = "EvolInstruct-Code-80k.json"

hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset", cache_dir="/root/autodl-tmp/datasets")
