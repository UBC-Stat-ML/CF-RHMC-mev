# MCMC base packages
using AbstractMCMC, MCMCChains, MCMCLogDensityProblems, LogDensityProblems
using AdvancedHMC, Pigeons
import Pigeons: @auto


# Data storing packages
using DataFrames, CSV, MAT

# Stat packages
using Distributions, SplittableRandoms, HypothesisTests, ExactOptimalTransport, StatsBase, Random, Statistics


# Linear algebra packages
using LinearRegression, LinearAlgebra, FillArrays

# AD packages
using ADTypes, Enzyme, Zygote, LogDensityProblemsAD, FiniteDiff, ForwardDiff


# Miscellanious packages
using BenchmarkTools, Optim, FFTW, LogExpFunctions, Match

# plotting packages
using Plots, StatsPlots


###############################################################################
# Mixing metrics
###############################################################################

function compute_ess(chain; degree = 2) 
   if degree == 1
      ess_df = ess(chain; autocov_method = FFTAutocovMethod(), maxlag = typemax(Int))
      if isa(chain,Chains)
         result = ess_df.nt.ess
      else
         result = [ess_df]
      end
   else
      ess_df = ess(Chains(Array(chain).^degree); autocov_method = FFTAutocovMethod(), maxlag = typemax(Int))
      result = ess_df.nt.ess
   end
   if minimum(result) < 100
       @warn "Low ESS: $(minimum(result))"
   end
   return result
end

function compute_wass(samples, true_samples)
   samples_freqs = countmap(samples)
   samples_vals = Float64.(keys(samples_freqs))
   samples_probs = Float64.(values(samples_freqs))
   samples_probs = samples_probs/sum(samples_probs)
   M = length(true_samples)
   return wasserstein(DiscreteNonParametric(samples_vals,samples_probs),
            DiscreteNonParametric(true_samples,fill(1 / M, M)); p = 2)
end

function compute_ks(samples, true_samples)
   return HypothesisTests.ksstats(samples,true_samples)[3]
end

###############################################################################
# summary stat from Chains
###############################################################################

function get_chain_quantiles(chain, idx)
   quantile_mat = Matrix{Float64}(undef, length(chain), 5)
   for i in 1:length(chain)
      quantile_mat[i,:] = quantile(chain[1:i,idx,:])
   end
   return quantile_mat
end

###############################################################################
# loading files
###############################################################################

function base_dir()
   base_folder = dirname(Base.active_project())
   return base_folder
end

function get_summary_df(experiment::String, folder = "output", file = "summary.csv")
   base_folder = dirname(base_dir())
   csv_path    = joinpath(base_folder, "deliverables", experiment, folder, file)
   return DataFrame(CSV.File(csv_path))
end


###############################################################################
# concatenation with index reset for Chains
###############################################################################

function auto_vcat(a::Chains,b::Chains)
   len = length(a) + length(b)
   return setrange(vcat(a,setrange(b,range(a)[end] .+ range(b) .- range(b)[1] .+ 1)), 1:len)
end

###############################################################################
# Bundle transition output from AdvancedHMC sample into Chains
###############################################################################

# for output from MCMCChains
function bundle_samples(
   ts::Vector{<:AdvancedHMC.Transition};
   param_names=missing,
   kwargs...,
)
   # state dimension
   d = length(ts[1].z.θ)

   # Turn all the transitions into a vector-of-vectors.
   vals = copy(reduce(hcat, [vcat(t.z.θ, vcat(values(t.stat)...)) for t in ts])')

   # Check if we received any parameter names.
   if ismissing(param_names)
      param_names = [Symbol("Parameter $i") for i in 1:d]
   end

   # Add the log density field to the parameter names.
   internal_params = vcat(keys(ts[1].stat)...)
   push!(param_names, internal_params...)

   # Bundle everything up and return a Chains struct.
   return Chains(vals, param_names, (internals=internal_params,))
end

# for output from AdvancedHMC
function bundle_samples(
   samples, stats;
   param_names=missing,
   kwargs...,
)
   # state dimension
   d = length(samples[1])
   n = length(samples)

   # Turn all the transitions into a vector-of-vectors.
   vals = copy(reduce(hcat, [vcat(samples[i], vcat(values(stats[i])...)) for i in 1:n])')

   # Check if we received any parameter names.
   if ismissing(param_names)
      param_names = [Symbol("Parameter $i") for i in 1:d]
   end

   # Add the log density field to the parameter names.
   internal_params = vcat(keys(stats[1])...)
   push!(param_names, internal_params...)

   # Bundle everything up and return a Chains struct.
   return Chains(vals, param_names, (internals=internal_params,))
end


###############################################################################
# fixed point iter utils
###############################################################################

function fixedpoint_iter(f, q_full, p_full, q_prev, p_prev; rel_tol = 0.01, max_iter = 50)
   f(q_full, p_full)
   j = 1

   # store closest solution to fp problem
   sol_change = max(maximum(abs.(q_full - q_prev)), maximum(abs.(p_full - p_prev)))

   copy!(q_prev,q_full)
   copy!(p_prev,p_full)

   while ((j<max_iter) & (sol_change > rel_tol)) | (j<6)
      f(q_full, p_full)
      j += 1

      sol_change = max(maximum(abs.(q_full - q_prev)), maximum(abs.(p_full - p_prev)))

      copy!(q_prev,q_full)
      copy!(p_prev,p_full)
   end

   # count traj shooting to Inf as integrator fail
   if isnan(sol_change)
      j = copy(max_iter)
   end

   # return number of fp iter and if fp failed
   return j, Int((j == max_iter))
end
