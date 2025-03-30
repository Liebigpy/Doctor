import sys
import os
import numpy as np
import wandb
import torch
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger
from research.env.d4rl_ds import get_datasets
from research.tokenizers.continuous import ContinuousTokenizer
from research.tokenizers.base import TokenizerManager
from research.algo.utils import set_seed_everywhere, configure_optimizers
from research.buffer.d4rl_buffer import D4RLTrajectoryBuffer
from research.buffer.d4rl_buffer import SequenceDataset
from research.models.doctor_model_v4 import Doctor_Transformer
from research.algo.masks import MaskType, create_random_autoregressize_mask, create_random_masks
from research.algo.doctor_trainer import Doc_Trainer
from research.algo.utils import get_ckpt_path_from_folder

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"

def _schedule1(step):
    warmup_steps = 40000
    num_train_steps = 280010
    if step < warmup_steps:
        return step / warmup_steps
    # then cosine decay
    step = step - warmup_steps
    return 0.5 * (
            1 + np.cos(step / (num_train_steps - warmup_steps) * np.pi)
    )

class Model_Config:
    n_embd: int = 512
    n_head: int = 4
    n_enc_layer: int = 2
    n_dec_layer: int = 1
    dropout: float = 0.1
    embd_pdrop: float = 0
    resid_pdrop: float = 0
    attn_pdrop: float = 0
    norm: str = "none"
    loss: str = "total"
    reduce_use_sum: bool = False
    loss_keys: Optional[List[str]] = None
    latent_dim: Optional[int] = None
    use_masked_loss: bool = False

def main():

    cli_parser = argparse.ArgumentParser(description="run experiment")
    cli_parser.add_argument(
        '--path',
        type=str,
        default='/your_path/Doctor/research/config/doctor.py',
        help='path to config file',
    )
    cli_args, unknown = cli_parser.parse_known_args()
    config_ = parse_args(cli_args.path)
    # config_ = parse_args(path)

    # print('arg',config_)
    exp_name = "_".join([config_.task, "seed" + str(config_.seed)])

    logger = CompositeLogger(log_dir=f"../log/Doctor/{config_.name}", name=exp_name, logger_config={
        "CsvLogger": {"config": config_},
        "WandbLogger": {"config": config_, "settings": wandb.Settings(_disable_stats=True), **config_.wandb}
    }, activate=not config_.debug)
    setup(config_, logger)

    set_seed_everywhere(config_.seed)

    # ================ import dataset & create buffers and loaders ===================== #

    train_d, val_d, env = get_datasets(config_.task, train_val_split=0.97)
    config_.obs_shape = np.prod(env.observation_space.shape)
    config_.action_shape = np.prod(env.action_space.shape)
    offline_trajectory_buffer = D4RLTrajectoryBuffer(train_d, seq_len=config_.seq_len, discount=config_.discount, env=env)

    valid_buffer = SequenceDataset(val_d,seq_len=config_.seq_len,discount = config_.discount)
    val_sampler = torch.utils.data.SequentialSampler(valid_buffer)
    val_loader = DataLoader(valid_buffer, shuffle=False, batch_size=config_.train_batch_size, num_workers=1, sampler=val_sampler, )
    vis_batch = next(iter(val_loader))
    #print(vis_batch)
    # config_.reward_scale = offline_trajectory_buffer.reward_scale
    config_.Max_return = offline_trajectory_buffer.max_return
    #print("Reward_scale vs Max_return in dataset",config_.reward_scale,config_.Max_return)
    print("=======================================")

    # ================ create tokenizers ===================== #
    tokenizers = { k: ContinuousTokenizer.create(k, offline_trajectory_buffer) for k in config_.tokenizers}
    tokenizer_manager = TokenizerManager(tokenizers).to(device)
    discrete_map: Dict[str, bool] = {}
    for k, v in tokenizers.items():
        discrete_map[k] = v.discrete

    tokenized = tokenizer_manager.encode(vis_batch)
    data_shapes = {}
    for k, v in tokenized.items():
        data_shapes[k] = v.shape[-2:]

    # ================ create models and masks===================== #

    model_config = Model_Config
    model = Doctor_Transformer(data_shapes, config_.seq_len, model_config)
    model.reward_scale = config_.reward_scale
    model.Max_return = config_.Max_return
    model.to(device)
    model.train()

    offline_optimizer = configure_optimizers(
        model,
        learning_rate=config_.learning_rate,
        weight_decay=config_.weight_decay,
        betas=(0.9, 0.999),  # following BERT
    )
    offline_scheduler = LambdaLR(offline_optimizer, lr_lambda=_schedule1)

    mask_functions_map = {MaskType.AUTO_MASK: lambda: create_random_autoregressize_mask(
        data_shapes, config_.mask_ratios, config_.seq_len, device, config_.mode_weights),}
    mask_functions = [mask_functions_map[MaskType[i]] for i in config_.mask_patterns]
    eval_masks = create_random_masks(data_shapes, config_.mask_ratios, config_.seq_len, device)


    # ================ load models if exist ===================== #
    eval_max = defaultdict(lambda: -np.inf)
    if config_.model_load_path is not None:
        ckpt_path = get_ckpt_path_from_folder(config_.model_load_path)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=device)
            print(f"Resuming from checkpoint: {ckpt_path}")
            step = ckpt["step"]
            model.load_state_dict(ckpt["model"])
            eval_max = ckpt["eval_max"]  # keep track of the max even after preempt
            print(f"starting from step={step}")
        else:
            print(f"No checkpoints found, starting from scratch.")

    # ================ create trainer and pretrain ===================== #
    trainer = Doc_Trainer(config_, model, mask_functions, discrete_map, logger, eval_masks,
                              offline_optimizer = offline_optimizer,
                              offline_scheduler = offline_scheduler,
                              eval_max = eval_max,
                              offline_buffer = offline_trajectory_buffer,
                              val_loader = val_loader,
                              tokenizer_manager = tokenizer_manager,
                              env = env,
                              rnd_model = None,
                              eval_num = 10
                              )
    trainer.pretrain()

if __name__ == '__main__':
    main()



























