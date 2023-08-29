import os
from modal import Image, Secret, Stub, method, gpu

def download_model_to_folder():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "google/flan-t5-xl",
        local_dir="/model",
        token=os.environ["HUGGINGFACE_TOKEN"],
    )

image = (
    Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install(
        "torch==2.0.1", index_url="https://download.pytorch.org/whl/cu118"
    )
    .pip_install(
        "hf_transfer",
        "transformers==4.26.0",
        "datasets==2.9.0",
        "accelerate==0.16.0",
        "evaluate==0.4.0",
        "deepspeed==0.8.0", 
        "ninja",
        "rouge-score", "nltk", "py7zr", "tensorboard"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # .env({"NCCL_P2P_DISABLE": "1"})
    .run_function(
        download_model_to_folder,
        secret=Secret.from_name("huggingface"),
        timeout=60 * 20,
    )
)

stub = Stub("nvlink-bench", image=image)
@stub.function(gpu=gpu.A100(count=4), secret=Secret.from_name("huggingface"))
def train():
    import os
    print("NCLL_P2P_DISABLE", os.getenv("NCCL_P2P_DISABLE"))
    print(os.system("nvidia-smi"))
    print(os.system("nvidia-smi nvlink -s"))
    print(os.system("nvidia-smi topo -m"))
    

@stub.local_entrypoint()
def main():
    train.remote()
 