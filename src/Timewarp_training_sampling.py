import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from bgflow.utils import as_numpy


from tqdm.auto import tqdm
from timewarp.utils.config_utils import (
    finalize_config,
    load_config_dict_in_subdir,
)
from timewarp.utils.training_utils import load_or_construct_model
from scipy.stats import special_ortho_group
from torch.utils.tensorboard import SummaryWriter
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

system = sys.argv[1]
assert system in ["CLN", "TRP"]


noise = float(sys.argv[2])
layers = int(sys.argv[3])
transformer_layers = int(sys.argv[4])
spacing = int(sys.argv[5])
repeats = spacing

if system == "CLN":
    npz_file = np.load("../data/charmm22star_cln_opt_data.npz")
    dim = 30
    lengthscales = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

elif system == "TRP":
    npz_file = np.load("../data/charmm22star_trp_opt_data.npz")
    dim = 60
    lengthscales = [0.5, 1.0, 2, 4, 8, 16]

data = torch.from_numpy(npz_file["positions"]).float()
data_conditioning = data
data_target = data
data_all = torch.cat([data_conditioning, data_target], dim=1).reshape(-1, 2 * dim)[
    ::spacing
]

torch.manual_seed(42)
idx = torch.randperm(data_all.shape[0])
data_all = data_all[idx]
data_train = data_all[:-1000]
data_val = data_all[-1000:]

# model
config = load_config_dict_in_subdir("timewarp/configs/kernel_transformer_nvp.yaml")
config = finalize_config(config)
config.model_config.custom_transformer_nvp_config.encoder_layer_config.lengthscales = (
    lengthscales
)

config.model_config.custom_transformer_nvp_config.num_coupling_layers = layers
config.model_config.custom_transformer_nvp_config.num_transformer_layers = (
    transformer_layers
)

model = load_or_construct_model(config).to(device)
dim = dim * 4
n_beads = 20
model.flow.atom_embedder = nn.Embedding(
    num_embeddings=n_beads,
    embedding_dim=config.model_config.custom_transformer_nvp_config.atom_embedding_dim,
).to(device)

# training
model = (
    f"{system}_Timewarp_l{layers}_tl{transformer_layers}_noise{noise}_spacing{spacing}"
)
PATH = f"../models/{model}"

batch_size = 512
epochs = 1000
data_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(data_val, batch_size=len(data_val), shuffle=True)
optim = torch.optim.Adam(model.parameters(), lr=5e-4)
losses = []
random_velocs = True
writer = SummaryWriter("logs/" + PATH)
try:
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    global_it = checkpoint["global_it"]
    print("Successfully loaded model")
except:
    print("Generated new model")
    epoch = 0
    global_it = 0
print(PATH, lengthscales)

for epoch in tqdm(range(epoch, epochs)):
    for batch in data_loader:
        optim.zero_grad()
        n_batch = batch.shape[0]
        batch = batch.view(n_batch, -1, 3)
        so = special_ortho_group(dim=3)
        rotation = so.rvs(n_batch).reshape(n_batch, 3, 3)
        rotation = torch.Tensor(rotation).to(batch)
        batch = torch.einsum("nmd, nde -> nme", batch, rotation)
        batch = batch @ rotation
        batch = batch.view(n_batch, -1)

        x_cond = batch[:, : dim // 4]
        x_cond = x_cond + noise * torch.randn_like(x_cond)

        x_targ = batch[:, dim // 4 :]
        if random_velocs:
            v_cond = torch.zeros_like(x_cond)
            v_targ = torch.randn_like(x_targ)
        else:
            raise NotImplementedError()
        nll = model(
            atom_types=torch.arange(0, n_beads, 1)
            .reshape(1, -1)
            .to(device, non_blocking=True)
            .repeat(n_batch, 1),
            x_coords=x_cond.to(device, non_blocking=True).reshape(-1, n_beads, 3),
            x_velocs=v_cond.to(device, non_blocking=True).reshape(-1, n_beads, 3),
            y_coords=x_targ.to(device, non_blocking=True).reshape(-1, n_beads, 3),
            y_velocs=v_targ.to(device, non_blocking=True).reshape(-1, n_beads, 3),
            adj_list=torch.tensor(
                [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]
            )
            .to(device, non_blocking=True)
            .repeat(n_batch, 1),
            edge_batch_idx=torch.arange(0, n_batch, 1, dtype=int)
            .to(device, non_blocking=True)
            .repeat_interleave(n_beads - 1),
            masked_elements=torch.zeros((1, n_beads), dtype=bool)
            .to(device, non_blocking=True)
            .repeat(n_batch, 1),
        )
        nll.backward()
        optim.step()
        losses.append(as_numpy(nll))
        writer.add_scalar("Loss/Train", nll, global_step=global_it)
        global_it += 1

    # eval
    with torch.no_grad():
        tqdm_val_data = tqdm(val_data_loader)
        for batch in tqdm_val_data:

            optim.zero_grad()
            n_batch = batch.shape[0]
            batch = batch.view(n_batch, -1, 3)
            so = special_ortho_group(dim=3)
            rotation = so.rvs(n_batch).reshape(n_batch, 3, 3)
            rotation = torch.Tensor(rotation).to(batch)
            batch = torch.einsum("nmd, nde -> nme", batch, rotation)
            batch = batch @ rotation
            batch = batch.view(n_batch, -1)

            x_cond = batch[:, : dim // 4]
            x_cond = x_cond + noise * torch.randn_like(x_cond)
            x_targ = batch[:, dim // 4 :]
            if random_velocs:
                v_cond = torch.zeros_like(x_cond)
                v_targ = torch.randn_like(x_targ)
            else:
                raise NotImplementedError()
            nll = model(
                atom_types=torch.arange(0, n_beads, 1)
                .reshape(1, -1)
                .to(device, non_blocking=True)
                .repeat(n_batch, 1),
                x_coords=x_cond.to(device, non_blocking=True).reshape(-1, n_beads, 3),
                x_velocs=v_cond.to(device, non_blocking=True).reshape(-1, n_beads, 3),
                y_coords=x_targ.to(device, non_blocking=True).reshape(-1, n_beads, 3),
                y_velocs=v_targ.to(device, non_blocking=True).reshape(-1, n_beads, 3),
                adj_list=torch.tensor(
                    [
                        [0, 1],
                        [1, 2],
                        [2, 3],
                        [3, 4],
                        [4, 5],
                        [5, 6],
                        [6, 7],
                        [7, 8],
                        [8, 9],
                    ]
                )
                .to(device, non_blocking=True)
                .repeat(n_batch, 1),
                edge_batch_idx=torch.arange(0, n_batch, 1, dtype=int)
                .to(device, non_blocking=True)
                .repeat_interleave(n_beads - 1),
                masked_elements=torch.zeros((1, n_beads), dtype=bool)
                .to(device, non_blocking=True)
                .repeat(n_batch, 1),
            )
        writer.add_scalar("Loss/Val", nll, global_step=global_it)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "epoch": epoch,
            "global_it": global_it,
        },
        PATH,
    )


# Force generation
def flow_sampler(x_cond, random_velocs=random_velocs):
    with torch.no_grad():
        if random_velocs:
            v_cond = torch.zeros_like(x_cond)
        y_coords, y_velocs = model.conditional_sample(
            atom_types=torch.arange(0, n_beads, 1)
            .reshape(1, -1)
            .to(device, non_blocking=True)
            .repeat(len(x_cond), 1),
            x_coords=x_cond.to(device, non_blocking=True).reshape(-1, n_beads, 3),
            x_velocs=v_cond.to(device, non_blocking=True).reshape(-1, n_beads, 3),
            adj_list=torch.tensor(pairs)
            .to(device, non_blocking=True)
            .repeat(len(x_cond), 1),
            edge_batch_idx=torch.arange(0, len(x_cond), 1, dtype=int)
            .to(device, non_blocking=True)
            .repeat_interleave(n_beads - 1),
            masked_elements=torch.zeros((1, n_beads), dtype=bool)
            .to(device, non_blocking=True)
            .repeat(len(x_cond), 1),
            num_samples=1,
        )

    samples = torch.cat(
        [
            x_cond.reshape(-1, n_beads * 3),
            v_cond.reshape(-1, n_beads * 3),
            y_coords.reshape(-1, n_beads * 3),
            y_velocs.reshape(-1, n_beads * 3),
        ],
        dim=-1,
    )
    return samples


def compute_force(x_cond, xv, random_velocs=random_velocs, device=device):
    # here we take the grad wrt positions and noise
    xv.requires_grad = True
    if random_velocs:
        v_cond = torch.zeros_like(x_cond)

    xv = xv.to(device, non_blocking=True).reshape(-1, n_beads * 2, 3)
    nll = -model.log_likelihood(
        atom_types=torch.arange(0, n_beads, 1)
        .reshape(1, -1)
        .to(device, non_blocking=True)
        .repeat(len(x_cond), 1),
        x_coords=x_cond.to(device, non_blocking=True).reshape(-1, n_beads, 3),
        x_velocs=v_cond.to(device, non_blocking=True).reshape(-1, n_beads, 3),
        y_coords=xv[:, :n_beads],
        y_velocs=xv[:, n_beads:],
        adj_list=torch.tensor(pairs)
        .to(device, non_blocking=True)
        .repeat(len(x_cond), 1),
        edge_batch_idx=torch.arange(0, len(x_cond), 1, dtype=int)
        .to(device, non_blocking=True)
        .repeat_interleave(n_beads - 1),
        masked_elements=torch.zeros((1, n_beads), dtype=bool)
        .to(device, non_blocking=True)
        .repeat(len(x_cond), 1),
    )
    force = torch.autograd.grad(-nll.sum(), xv)[0].reshape(-1, n_beads * 3 * 2)
    return force, nll


force_from_data = False
print("force_from_data:", force_from_data)

n_batches = 10000
forces = []
nll = []
positions = []
spacing = 10000

for i in range(n_batches):
    conditioning = (
        data_train[i::spacing, : dim // 4].reshape(-1, dim // 4).to(device)
    ).repeat_interleave(repeats, dim=0)
    conditioning = conditioning + noise * torch.randn_like(conditioning)

    conditioning_x_i = flow_sampler(conditioning)

    forces_i, nll_i = compute_force(
        conditioning_x_i[:, : dim // 4], conditioning_x_i[:, dim // 2 :]
    )
    forces.append(forces_i.detach().cpu())
    nll.append(nll_i.detach().cpu())
    positions.append(conditioning_x_i[:, dim // 2 : -dim // 4].detach().cpu())
    if i % 1000 == 0:
        print(f"{i}/{n_batches}")


forces = np.concatenate(forces, axis=0)
positions = np.concatenate(positions, axis=0)

np.savez(
    (f"../results_data/{model}_multiple{repeats}.npz"),
    forces=as_numpy(forces),
    positions=as_numpy(positions),
)
