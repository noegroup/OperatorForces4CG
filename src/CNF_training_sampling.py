import torch
import numpy as np

from bgflow.utils import (
    remove_mean,
    IndexBatchIterator,
    as_numpy,
)
from bgflow import (
    DiffEqFlow,
    BoltzmannGenerator,
    MeanFreeNormalDistribution,
)
from cnf_models import EGNN_dynamics_AD2_cat
from bgflow import (
    BlackBoxDynamics,
)
from torchdyn.core import NeuralODE
from utils import BruteForceEstimatorFast
import signal

import tqdm

import sys

system = sys.argv[1]
assert system in ["CLN", "TRP"]

noise_std = float(sys.argv[2])
noise_prior = float(sys.argv[3])
sigma = float(sys.argv[4])
spacing = float(sys.argv[5])

training = True
sampling = True

model_name = f"{system}_CNF-noise{noise_std}-noiseprior{noise_prior}-sigma{sigma}-spacing{spacing}"

print(model_name)

# Include the data here
if system == "CLN":
    npz_file = np.load("../data/charmm22star_cln_opt_data.npz")
    n_beads = 10
elif system == "TRP":
    npz_file = np.load("../data/charmm22star_trp_opt_data.npz")
    n_beads = 20

data_1Mx = npz_file["positions"]
forces_1Mx = npz_file["forces"]

n_particles = n_beads
n_dimensions = 3
dim = n_beads * n_dimensions

h_initial = torch.nn.functional.one_hot(torch.arange(n_beads))

scaling = 1
all_data = (
    remove_mean(torch.from_numpy(data_1Mx), n_beads, n_dimensions)
    .reshape(-1, dim)
    .float()
    * scaling
)
# split data
data = all_data[:-1000]
data_holdout = all_data[-1000:]

n_data = len(data) // spacing
np.random.seed(0)
idx = np.random.choice(np.arange(len(data)), n_data, replace=False)
data_smaller = data[idx]

# now set up a prior
prior = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
prior_cpu = MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False)

brute_force_estimator_fast = BruteForceEstimatorFast()
net_dynamics = EGNN_dynamics_AD2_cat(
    n_particles=n_particles,
    device="cuda",
    n_dimension=dim // n_particles,
    h_initial=h_initial,
    hidden_nf=64,
    act_fn=torch.nn.SiLU(),
    n_layers=5,
    recurrent=True,
    tanh=True,
    attention=True,
    condition_time=True,
    mode="egnn_dynamics",
    agg="sum",
)
bb_dynamics = BlackBoxDynamics(
    dynamics_function=net_dynamics, divergence_estimator=brute_force_estimator_fast
)
flow = DiffEqFlow(dynamics=bb_dynamics)
bg = BoltzmannGenerator(prior, flow, prior).cuda()
flow._use_checkpoints = True

# Anode options
options = {"Nt": 20, "method": "RK4"}
flow._kwargs = options

optim = torch.optim.Adam(bg.parameters(), lr=5e-4)
n_epochs = 500 * spacing
n_batch = 512
batch_iter = IndexBatchIterator(len(data_smaller), n_batch)
bath_iter_val = IndexBatchIterator(len(data_holdout), n_batch * 4)
if training:
    for epoch in tqdm.tqdm(range(n_epochs)):
        with torch.no_grad():
            val_loss = 0
            for it, idx in enumerate(bath_iter_val):
                x_cond = data_holdout[idx].cuda()
                batchsize = x_cond.shape[0]
                x1 = prior_cpu.sample(batchsize).cuda() * noise_std

                t = torch.rand(batchsize, 1).cuda()
                x0 = prior_cpu.sample(batchsize).cuda() * noise_prior

                # calculate regression loss
                mu_t = x0 * (1 - t) + x1 * t
                sigma_t = sigma
                noise = prior.sample(batchsize)
                x = mu_t + sigma_t * noise
                ut = x1 - x0
                vt = bg.flow._dynamics._dynamics._dynamics_function(t, x + x_cond - x1)
                val_loss += torch.mean((vt - ut) ** 2)

            val_loss = val_loss / len(data_holdout) * n_batch * 4
            print(val_loss)

        if epoch == 250 * spacing:
            for g in optim.param_groups:
                g["lr"] = 5e-5
        for it, idx in enumerate(batch_iter):
            optim.zero_grad()

            x_cond = data_smaller[idx].cuda()
            batchsize = x_cond.shape[0]
            x1 = prior_cpu.sample(batchsize).cuda() * noise_std

            t = torch.rand(batchsize, 1).cuda()
            x0 = prior_cpu.sample(batchsize).cuda() * noise_prior

            # calculate regression loss
            mu_t = x0 * (1 - t) + x1 * t
            sigma_t = sigma
            noise = prior.sample(batchsize)
            x = (
                mu_t + sigma_t * noise
            )  # noise_sampler.sample(batchsize) * torch.tensor([[1.,0.,-1,0]]).to(x0)#*0.7
            ut = x1 - x0
            vt = bg.flow._dynamics._dynamics._dynamics_function(t, x + x_cond - x1)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optim.step()

PATH_last = f"../models/{model_name}"

if training:
    torch.save(
        {
            "model_state_dict": bg.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
        },
        PATH_last,
    )
else:
    checkpoint = torch.load(PATH_last)
    bg.load_state_dict(checkpoint["model_state_dict"])
    print(f"model loaded: {model_name}")

# use OTD in the evaluation process
bb_dynamics._divergence_estimator = brute_force_estimator_fast
bg.flow._integrator_atol = 1e-4
bg.flow._integrator_rtol = 1e-4
flow._use_checkpoints = False
flow._kwargs = {}

repeats = spacing
n_samples = 10000 // repeats
n_sample_batches = len(data_smaller) // n_samples

node = NeuralODE(
    net_dynamics, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
)

latent_np = np.empty(shape=(0))
samples_np = np.empty(shape=(0))
conditioning_np = np.empty(shape=(0))

if sampling:
    for i in tqdm.tqdm(range(n_sample_batches)):
        with torch.no_grad():
            x1 = (
                data_smaller[i * n_samples : (i + 1) * n_samples]
                .repeat_interleave(repeats, dim=0)
                .cuda()
            )
            conditioning = x1 + prior_cpu.sample(x1.shape[0]).cuda() * noise_std
            latent = conditioning + prior_cpu.sample(x1.shape[0]).cuda() * noise_prior
            traj = node.trajectory(
                latent,
                t_span=torch.linspace(0, 1, 100),
            )
            samples = traj[-1]

            latent_np = np.append(latent_np, latent.detach().cpu().numpy())
            conditioning_np = np.append(
                conditioning_np, conditioning.detach().cpu().numpy()
            )

            samples_np = np.append(samples_np, samples.detach().cpu().numpy())

    latent_np = latent_np.reshape(-1, dim)
    samples_np = samples_np.reshape(-1, dim)
    conditioning_np = conditioning_np.reshape(-1, dim)

if sampling:
    np.savez(
        f"../results_data/{model_name}_sampling",
        latent_np=latent_np,
        samples_np=samples_np,
        conditioning_np=conditioning_np,
    )
else:
    npz = np.load(f"../results_data/{model_name}_sampling.npz")
    samples_np = npz["samples_np"]
    latent_np = npz["latent_np"]
    conditioning_np = npz["conditioning_np"]
    print("sampling data loaded")


def handler(signum, frame):
    raise TimeoutError("Function call took too long")


# some constants
kb = 1.380649 * 10 ** (-23)
kbt = kb * 350
kcal = 4184  # J
mol = 6.02214076 * 10**23

forces_np = np.empty(shape=(0))
skipped_i = []

for i in tqdm.tqdm(range(0, n_sample_batches)):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(500)  # Timeout set to 500 seconds

    try:
        samples_x = (
            torch.from_numpy(
                samples_np[i * n_samples * spacing : (i + 1) * n_samples * spacing]
            )
            .cuda()
            .float()
        )
        conditioning_x = (
            torch.from_numpy(
                conditioning_np[i * n_samples * spacing : (i + 1) * n_samples * spacing]
            )
            .cuda()
            .float()
        )
        sampled_noise = samples_x - conditioning_x

        sampled_noise.requires_grad = True
        latent_samples_plus_conditioning, latent_dlogp = flow(
            sampled_noise + conditioning_x, inverse=True
        )
        latent_samples = latent_samples_plus_conditioning - conditioning_x
        nll = -latent_dlogp + prior.energy(latent_samples / noise_prior)
        flow_forces = torch.autograd.grad(-nll.sum(), sampled_noise)[0]
        forces_np = np.append(forces_np, as_numpy(flow_forces))
    except TimeoutError:
        print("Skipping iteration, function call took too long")
        skipped_i.append(i)
        forces_np = np.append(forces_np, as_numpy(torch.zeros_like(samples_x)))
        continue  # Skip to the next iteration
    finally:
        signal.alarm(0)
    if i % 10 == 5:
        save_name = f"../results_data/{model_name}_repeats{repeats}.npz"

        np.savez(
            save_name,
            positions=samples_np[: len(forces_np)].reshape(-1, n_beads, 3),
            forces=forces_np.reshape(-1, n_beads, 3) * (mol / kcal * kbt),
        )


save_name = f"../results_data/{model_name}_repeats{repeats}.npz"

np.savez(
    save_name,
    positions=samples_np.reshape(-1, n_beads, 3)[: len(forces_np)],
    forces=forces_np.reshape(-1, n_beads, 3) * (mol / kcal * kbt),
)
