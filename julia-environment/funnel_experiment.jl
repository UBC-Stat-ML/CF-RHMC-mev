include("src/sampling_functions.jl")
include("src/plotting_functions.jl")

#### 100D case

function funnel_summary(algo; exclude_failure = false)
    df_funnel1 = get_summary_df("funnel_$(algo)","output","first_coord.csv")
    df_funnel_time = get_summary_df("funnel_$(algo)","output","timing.csv")
    df_funnel_steps = get_summary_df("funnel_$(algo)","output","n_steps.csv")

    idx = df_funnel1.sampler .== algo
    samples_vec = Array(df_funnel1[idx,6:end])' # N x M matrix
    time_vec = mean(Array(df_funnel_time[idx,6:end])'; dims = 2) # N length vec

    if exclude_failure
        idx_good = []
        for i in 1:size(samples_vec,2)
            if samples_vec[1,i] != samples_vec[end,i]
                push!(idx_good,i)
            end
        end
        samples_vec = samples_vec[:,idx_good]
    end

    N, M = size(samples_vec)
    true_sample_vec = rand(SplittableRandom(2423),Normal(0,3),M)
    wass2_vec = []
    for i in 1:N
        push!(wass2_vec,wasserstein(DiscreteNonParametric(samples_vec[i,:],fill(1 / M, M)),
            DiscreteNonParametric(true_sample_vec,fill(1 / M, M)); p = 2))
    end

    return wass2_vec, samples_vec, time_vec, M
end

wass2_hmc, samples_hmc, time_hmc, nostuck_hmc = funnel_summary("hmc_unit")
wass2_nuts, samples_nuts, time_nuts, nostuck_nuts = funnel_summary("nuts_unit")
wass2_hmc_diag, samples_hmc_diag, time_hmc_diag, nostuck_hmc_diag = funnel_summary("hmc_diag")
wass2_nuts_diag, samples_nuts_diag, time_nuts_diag, nostuck_nuts_diag = funnel_summary("nuts_diag")
wass2_cgrhmc, samples_cgrhmc, time_cgrhmc, nostuck_cgrhmc = funnel_summary("cgrhmc")
wass2_cgrnuts, samples_cgrnuts, time_cgrnuts, nostuck_cgrnuts = funnel_summary("cgrnuts")
wass2_rhmc, samples_rhmc, time_rhmc, nostuck_rhmc = funnel_summary("rhmc")
wass2_rnuts, samples_rnuts, time_rnuts, nostuck_rnuts = funnel_summary("rnuts")

plot(time_cgrhmc, wass2_cgrhmc, label="CG-RHMC", color = :royalblue1, #aspect_ratio = 4,
    xlabel="Time (s)", ylabel="W₂ distance", xlimit=[0,20], ylimit=[0.5,2.5])
plot!(time_cgrnuts, wass2_cgrnuts, label="CG-RNUTS", color = :steelblue1)
plot!(time_rhmc,wass2_rhmc, label="Standard RHMC", color = :royalblue1, linestyle = :dash)
plot!(time_rnuts, wass2_rnuts, label="Standard RNUTS", color = :steelblue1, linestyle = :dash)
scatter!([13],[wass2_rnuts[1]],marker = :x,markersize = 7,label="")
plot!(time_hmc, wass2_hmc, label="Unit HMC")
plot!(time_nuts, wass2_nuts, label="Unit NUTS", color =:black)
plot!(time_hmc_diag, wass2_hmc_diag, label="Diagonal HMC")
plot!(time_nuts_diag, wass2_nuts_diag, label="Diagonal NUTS")

using CairoMakie
f, ax, l = CairoMakie.lines(0..9, sin; axis = (; xgridvisible = false, ygridvisible = false))
CairoMakie.ylims!(ax, 0, 2.5)
CairoMakie.xlims!(ax, 0, 20)

CairoMakie.bracket!(pi/2, 1, 5pi/2, 1, offset = 5, text = "Period length", style = :square)

CairoMakie.bracket!(10, 1.8, 10, 1.3, text = "Non-riemannian\nmethods", orientation = :up,
    rotation = 0, align = (:left, :center), textoffset = 4, linewidth = 1, color = :red, textcolor = :red)

CairoMakie.bracket!(10, 0.6, 10, 0.2, text = "CG based\nRiemannian\nmethods", orientation = :up,
    rotation = 0, align = (:left, :center), textoffset = 4, linewidth = 1, color = :blue, textcolor = :blue)

f

save("funnel_annotation.png", f)


plot(100 .* time_rhmc, wass2_rhmc, label="RHMC", color = :royalblue1,
    xlabel="Time (s)", ylabel="W₂ distance", xlimit=[0,20])
plot!(100 .* time_rnuts, wass2_rnuts, label="RNUTS", color = :steelblue1)
plot!(time_hmc, wass2_hmc, label="Unit HMC")
plot!(time_nuts, wass2_nuts, label="Unit NUTS", color =:black)
plot!(time_hmc_diag, wass2_hmc_diag, label="Diagonal HMC")
plot!(time_nuts_diag, wass2_nuts_diag, label="Diagonal NUTS")

df_funnel1 = get_summary_df("funnel_convergence","output_rnuts","first_coord.csv")
samples_vec = Array(df_funnel1[idx,6:end])'

idx_good = []
idx_no_good = []
for i in 1:1000
    if samples_vec[1,i] != samples_vec[end,i]
        push!(idx_good,i)
    else
        push!(idx_no_good,i)
    end
end

samples_good = samples_vec[:,idx_good]

a = samples_good[1000,:]
density(a)