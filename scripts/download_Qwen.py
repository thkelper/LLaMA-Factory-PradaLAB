from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen-72B-Chat-Int4', cache_dir='/data/models/Qwen')
