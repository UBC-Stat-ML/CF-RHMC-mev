#######################################################################################
# real data glm experiment plots
#######################################################################################

include("src/sampling_functions.jl")
include("src/plotting_functions.jl")

target_seed = 675
target_name = :glm
target_specs = (;n = 10, prior_type = :modifiedhorseshoe, data_name = "prostate_small")
target_test = build_target(target_name, target_seed; target_specs...)


data_name = "prostate_small_10"
# data_name = "prostate_small"
df_NRPT = get_summary_df(data_name,"output","chain_NRPT.csv")
df_NUTS = get_summary_df(data_name,"output","chain_NUTS.csv")
df_RNUTS = get_summary_df(data_name,"output","chain_RNUTS.csv")

df_stats_NRPT = get_summary_df(data_name,"output","summary_NRPT.csv")
df_stats_NUTS = get_summary_df(data_name,"output","summary_NUTS.csv")
df_stats_RNUTS = get_summary_df(data_name,"output","summary_RNUTS.csv")

ks_nuts = Array(df_stats_NUTS[1,4:(end-7)])
ks_rnuts = Array(df_stats_RNUTS[1,4:(end-7)])

chain_NRPT = Chains(Array(df_NRPT[:,4:end]))
chain_NUTS = Chains(Array(df_NUTS[:,4:end]))
chain_RNUTS = Chains(Array(df_RNUTS[:,4:end]))

# NRPT correctness check plots
# plot restart rate
summary_NRPT = df_stats_NRPT[1:15,:]
plot(summary_NRPT.round, summary_NRPT.n_tempered_restarts ./ summary_NRPT.n_scans, 
label = "", marker = :circle, xlabel = "Round", ylabel = "Tempered restart rate")

# plot chain quantile estimates
m = 57
quantile_mat = get_chain_quantiles(chain_NRPT, m)
errorline(1:length(chain_NRPT), quantile_mat[:,2:4], errorstyle=:ribbon, fillalpha=0.3, 
label = "NRPT", xlabel = "Iteration", ylabel = "λ₇")


# plot KS stat result
glm_KS_scatter_plot()

# plot density result
m = 57
m = 102
density(chain_NRPT[:,m,:], label = "NRPT", lw = 2, linestyle = :dash, color = :black,
xtickfontsize=10, ytickfontsize=10, legendfontsize=11)
density!(chain_RNUTS[:,m,:], label = "CF-RNUTS", lw = 2, color = 1)
density!(chain_NUTS[:,m,:], label = "NUTS", linestyle = :dot, lw = 2, color = 2)

# pair plots
m = 36; k = 12
p = glm_pair_plots(chain_NRPT,chain_RNUTS,chain_NUTS,target_test,m,k; 
folder = data_name, fsize = 13, layout_style = :none)

m = 102; k = 1
p = glm_pair_plots(chain_NRPT,chain_RNUTS,chain_NUTS,target_test,m,k; 
folder = data_name, fsize = 13, layout_style = :none)

m = 74; k = 24
p = glm_pair_plots(chain_NRPT,chain_RNUTS,chain_NUTS,target_test,m,k; 
folder = data_name, fsize = 13, layout_style = :none)

m = 102; k = 7
p = glm_pair_plots(chain_NRPT,chain_RNUTS,chain_NUTS,target_test,m,k; 
folder = data_name, fsize = 13, layout_style = :none)

m = 91; k = 1
p = glm_pair_plots(chain_NRPT,chain_RNUTS,chain_NUTS,target_test,m,k; 
folder = data_name, fsize = 13, layout_style = :none)

m = 102; k = 19 
p = glm_pair_plots(chain_NRPT,chain_RNUTS,chain_NUTS,target_test,m,k; 
folder = data_name, fsize = 13, layout_style = :vertical)

################################################
# panel of glm scale experiment plots
################################################

glm_scale_plots()