# CF-RHMC mev repo

## Prerequisite

- Install Julia 1.11
- Install Java 11

## Replication of results

To replicate an experiment, use for example

```bash
./nextflow run glm_scale.nf
```

The scripts correspond to the following figures:
- `timing_comparison.nf` is used for Figure 3,
- scripts with format `funnel_*.nf` are used for Figure 4,
- `prostate_small_10.nf` and `prostate_small.nf` are used for Figures 1, 5, 9, 10, 17 and 18,
- `glm_scale.nf` is used for Figures 6, 11 and 12,
- `fixed_point_scale.nf` is for Figure 13 and 14.


## Function usage

Some key functions in the Julia code:

- Generate target distribution `build_target(...)`.
- Sampling interface `sample_chain(...)`.
- Trajectory simulation interface `generate_full_trajectories(...)`.

These functions perform data preprocessing before their main tasks. 

For example, from the root of this repo:
```
using Pkg 
Pkg.activate("julia-environment")
Pkg.instantiate()
include("julia-environment/src/sampling_functions.jl")

# specify target
target_seed = 675 # seed for random subsampling
target_name = :glm
target_specs = (;n = 10, prior_type = :modifiedhorseshoe, data_name = "prostate_small")
target = build_target(target_name, target_seed; target_specs...)

# specify sampler
sampler_seed = 3324
sampler_specs = (; metric_name = :diag_rie)

# sample chain
n_steps = 10000
chain = sample_chain(sampler_seed, target_seed, n_steps;
    sampler_specs , target_name, target_specs, return_chain = true);

# generate a trajectory
seed = 235
n_steps = 500
step_size = 0.1
z_vec, θ_mat, r_mat, energy_vec, time, h = generate_full_trajectory(target, :diag_rie, seed, n_steps, step_size);

```

