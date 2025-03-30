from UtilsRL.misc import NameSpace

# ============basic parameter============
seed = 0
task = "hopper-medium-replay-v2"
name = "training_log"

class wandb(NameSpace):
    entity = "your-organization"
    project = "doctor-release"
    mode = "offline"

# ============pretrain loops parameter============
debug = True
train_batch_size = 1024
num_pretrain_steps = 120010
eval_every = 5000
print_every = 5000
log_every = 5000

# ============data size parameter============
seq_len = 6
num_workers = 4
pin_memory = True
tokenizers = ["states", "actions", "returns", "rewards"]
loss_keys = ["states",  "returns"]
mask_patterns = ["AUTO_MASK"]

# ============training parameter============
critic_lr = 3e-4
tau = 0.005
iql_tau = 0.7

learning_rate = 0.0001
weight_decay = 0.005
mask_ratios = [0.5,0.6,0.7,0.8,0.85,0.9,0.95,1.0]
mode_weights = (0.0,0.0,1.0)
discount = 1.5
reward_scale = 1.0
beta = 3.0
# ============paths============
model_save_path = "../net_para/"
model_load_path = None


