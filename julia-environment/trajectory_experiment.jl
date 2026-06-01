include("src/trajectory_utils.jl")

#############################################################################
# CG vs standard trajectory
#############################################################################

target_name = :glm
target_specs = (;dim = 8, n = 3, prior_type = :normal, synthetic = true)

target_test = build_target(target_name, 535; target_specs...)

# get trajectory
seed = 6435
n_steps = 500
ss = 0.1

# CG-RHMC
z_vec_Diag, θ_mat_Diag, r_mat_Diag, energy_Diag, time_Diag, ns_diag, h_diag = generate_full_trajectory(target_test, :diag_rie, seed, n_steps, ss; cache = true, a = Inf);
# RHMC
z_vec_Diag2, θ_mat_Diag2, r_mat_Diag2, energy_Diag2, time_Diag2, ns_diag2, h_diag2 = generate_full_trajectory(target_test, :diag_rie, seed, n_steps, ss; cache = false, a = Inf);


# plot trajectory
# trajectory plot
k = 1; l = 500; n = 4; m = 1; fsize = 9;
plot(θ_mat_Diag2[k:l,m],θ_mat_Diag2[k:l,n], label = "Standard RHMC", lw = 2, color = 2, xlabel = "θ₁", ylabel = "θ₄")
scatter!(θ_mat_Diag[k:l,m],θ_mat_Diag[k:l,n], label = "CG-RHMC", markersize=2, color = 1)
scatter!(θ_mat_Diag[k:k,m],θ_mat_Diag[k:k,n], marker = :star, 
label = "Starting point", color = :yellow, markersize=8,
xtickfontsize=fsize, ytickfontsize=fsize, legendfontsize=fsize+2, xguidefontsize=fsize+5, yguidefontsize=fsize+5)

# phase space plot
k = 1; l = 500; n = 4; m = 3; fsize = 9;
plot(θ_mat_Diag2[k:l,m],r_mat_Diag2[k:l,m], label = "", lw = 2, color = 2, xlabel = "θ₃", ylabel = "p₃")
scatter!(θ_mat_Diag[k:l,m],r_mat_Diag[k:l,m], label = "", markersize=2, color = 1)
scatter!(θ_mat_Diag[k:k,m],r_mat_Diag[k:k,m], marker = :star, 
label = "", color = :yellow, markersize=8,
xtickfontsize=fsize, ytickfontsize=fsize, legendfontsize=fsize+2, xguidefontsize=fsize+5, yguidefontsize=fsize+5)


#############################################################################
# mass metric regularization comparison
#############################################################################
target_name = :funnel
target_specs = (;dim = 100, b = 0.5)

target_test = build_target(target_name, 535; target_specs...)


target_seed = 234
sampler_seed = 1275 # stuck outside funnel
sampler_seed = 234 # stuck inside funnel
n = 10000

# w/o regularization
rnuts_specs = (; metric_name = :diag_rie, acc_rate = 0.8, a = Inf)
chain1, runtime1, adapttime1, n_steps1, h1 = sample_chain(sampler_seed, target_seed, n;
    sampler_specs = rnuts_specs, target_name, target_specs, return_chain = true, MINIMUM_REQUIRED_ESS = 0.0)

# w/ regularization
rnuts_specs = (; metric_name = :diag_rie, acc_rate = 0.8, a = 100.0)
chain2, runtime2, adapttime2, n_steps2, h2 = sample_chain(sampler_seed, target_seed, n;
    sampler_specs = rnuts_specs, target_name, target_specs, return_chain = true, MINIMUM_REQUIRED_ESS = 0.0)

# plot results
# chain trace plots
plot(vec(chain1[(n+1):end,1,1]), xlabel = "Iteration", ylabel = "Sample value", label = "", color=2, ylims = [-12,12])
plot(vec(chain2[(n+1):end,1,1]), xlabel = "Iteration", ylabel = "Sample value", label = "", ylims = [-12,12])

# density comparison
x = -10.0:0.001:10.0
density(vec(chain1[(n+1):end,1,1]), label = "Identity")
density!(vec(chain2[(n+1):end,1,1]), label = "Softabs")
plot!(x, pdf(Normal(0,3),x), linestyle = :dash, label = "Truth")
