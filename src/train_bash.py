from llmtuner import run_exp
import wandb
from datetime import datetime
# wandb.init(project="DoRA_commonsense", entity="prada-lab")

# TODO auto name 
# wandb.init(name=f"llamapro_expand2_Llama_8B_Instruct_lr9e-4_math10k_pbs1_ga16_{str(datetime.now().strftime('%Y.%m.%d-%H.%M.%S'))}", project="llama pro", entity="prada-lab")


def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
