from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen1.5-72B-Chat', cache_dir='/data/models/Qwen')
