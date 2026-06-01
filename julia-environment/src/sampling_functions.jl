include("utils.jl")
include("glm_utils.jl")
include("targets.jl")
include("Riemannian.jl")
include("metric_utils.jl")
include("hamiltonian_utils.jl")
include("sample_utils.jl")
include("EnzymeAD_utils.jl")
include("ZygoteAD_utils.jl")
include("pigeons_utils.jl")
include("trajectory_utils.jl")


#################################################################
# sampling function
#################################################################
function sample_hmc(ℓπ, N, seed = 1, burnin = 0;
      nuts_id = true, # use nuts or hmc
      multinomial_TS = false, # propose enpoint or random point on trajectory
      L = 20, # trajectory length
      adapt_mode = :both, # 4 mode: :none, :step_size, :metric, :both
      metric_name = :unit,
      kinetic_name = :gaussian,
      a = 5.0, # softabs smoothness param
      inv_mass_mat = nothing, # initial inverse mass matrix for :diagonal, :dense metric
      step_size = 1.0,
      integrator_name = nothing,
      acc_rate = 0.8,
      initstate = nothing,
      cache = true, 
      relativistic_params = [0.2, 4.0],
      max_fp = 50,
      fp_tol = 0.01,
   )
   rng = SplittableRandom(seed)
   if isnothing(initstate)
      if isa(ℓπ, LogisticRegression) 
         if isa(ℓπ.prior, HorseshoePrior)
            initstate = vcat(rand(rng,Int(floor(ℓπ.dim / 2))) .* 4 .- 2,
                  exp.(rand(rng,Int(ℓπ.dim / 2)) .* 4 .- 2))
         else
            initstate = rand(rng,ℓπ.dim) .* 4 .- 2
         end
      else
         initstate = rand(rng,ℓπ.dim) .* 4 .- 2
      end
   end

   kernel = rhmc_kernel(ℓπ.dim, custom_integrator(metric_name, integrator_name, step_size; max_fp, fp_tol), L, nuts_id, multinomial_TS)

   if a == Inf
      map = IdentityMap()
   else
      map = SoftAbsMap(a)
   end
   metric = create_metric(metric_name,ℓπ,ℓπ.dim;cache,map,a,initstate,inv_mass_mat)
   if kinetic_name == :gaussian
      h = Hamiltonian(metric, ℓπ, AutoEnzyme())
   elseif kinetic_name == :relativistic
      kine = RelativisticKinetic(relativistic_params...)
      h = relHamiltonian(metric, kine, ℓπ, AutoEnzyme())
   end

   if adapt_mode == :none
      adaptor = NoAdaptation()
   elseif (adapt_mode == :step_size) | (metric_name == :diag_rie)
      adaptor = StepSizeAdaptor(acc_rate,kernel.τ.integrator.ϵ)
   elseif adapt_mode == :metric
      adaptor = MassMatrixAdaptor(metric)
   elseif adapt_mode == :both
      mma = MassMatrixAdaptor(metric)
      ssa = StepSizeAdaptor(acc_rate,kernel.τ.integrator.ϵ)
      adaptor = StanHMCAdaptor(mma, ssa)
   end

   # begin sampling
   result = @timed sample(
      rng, h, kernel, initstate, N, adaptor, burnin; progress=false
   )
   
   samples, stats, h = result.value

   chain = bundle_samples(samples,stats)

   if (metric_name == :diag_rie) | (metric_name == :coord_rie)
      target_evals = h.metric.fp_count[1]
   else
      target_evals = sum(chain[:n_steps])
   end


   return chain, result.time, target_evals, h
end

# function to set integrator for kernel
function custom_integrator(metric_name, integrator_name, step_size; max_fp = 50, fp_tol = 0.01)
   if isnothing(integrator_name)
      return if (metric_name == :diag_rie) | (metric_name == :dense_rie)
         ImplicitMidpoint(step_size, max_fp, fp_tol)
      else
         Leapfrog(step_size)
      end
   elseif integrator_name == :im
      return ImplicitMidpoint(step_size, max_fp, fp_tol)
   elseif integrator_name == :glf
      return if metric_name == :dense_rie
         GeneralizedLeapfrog(step_size, 6)
      else
         error("integrator not applicable to current metric")
      end
   elseif integrator_name == :lf
      return if (metric_name == :unit) | (metric_name == :diagonal) | (metric_name == :dense)
         Leapfrog(step_size)
      else
         error("integrator not applicable to current metric")
      end
   end
end

#################################################################
# HMC kernel and sampler generator
#################################################################

function rhmc_kernel(d, integrator, n_leapfrog, nuts = false, multinomial_TS = false)
   if nuts
      return HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
   else
      if multinomial_TS
         return HMCKernel(Trajectory{MultinomialTS}(integrator, FixedNSteps(n_leapfrog)))
      else
         return HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(n_leapfrog)))
      end
   end
end


#################################################################
# target builder function
#################################################################

function build_target(target_name, seed; target_specs...)
   if target_name == :logistic_normal
      target = LogisticNormalTarget(seed; target_specs...)
   elseif target_name == :normal
      target = NormalTargetDensity(seed; target_specs...)
   elseif target_name == :logistic
      target = LogisticTargetDensity(seed; target_specs...)
   elseif target_name == :log_gamma
      target = LogGammaTarget(seed; target_specs...)
   elseif target_name == :cauchy
      target = CauchyTarget(seed; target_specs...)   
   elseif target_name == :glm
      target = glm_target(seed; target_specs...)
   elseif target_name == :funnel
      target = FunnelTarget(seed;target_specs...)
   elseif target_name == :gcpp
      target = gcpp_target(seed;target_specs...)
   elseif target_name == :sv
      target = SVtarget(seed;target_specs...)
   elseif target_name == :normal_mixture
      target = NormalMixture(seed;target_specs...)
   else
      error("undefined target")
   end
end

#################################################################
# general sampling function
#################################################################

function sample_chain(sampler_seed, target_seed, n_samples;
      n_rounds = nothing,
      alotted_adapt_time = nothing,
      MINIMUM_REQUIRED_ESS = 0,
      sampler_name = :NUTS,
      sampler_specs = (;),
      target_name = :logistic_normal,
      target_specs = (;),
      return_chain = false
   )

   # build target
   target = build_target(target_name, target_seed; target_specs...)

   @info "Running $sampler_name for a $target_name target with $(target_specs)"
   @info "Sampler specs: $(sampler_specs)"
   @info "Running 2 sample chain for code compilation"
   if sampler_name == :NUTS
      chain, time, target_evals, h = sample_hmc(target, 2, sampler_seed; sampler_specs...)
   else
      error("Sampler not defined")
   end

   @info "Running now..."
   if (isnothing(alotted_adapt_time) & isnothing(n_rounds)) # ess based sampling
      if sampler_name == :NUTS
         chain, time, target_evals, h = sample_hmc(target, 2*n_samples, sampler_seed, n_samples; sampler_specs...);
      else
         error("Sampler not defined")
      end
      adapttime = chain[:current_time][n_samples] - chain[:current_time][1]
      runtime = chain[:current_time][end] - chain[:current_time][n_samples + 1]
      chain_len = copy(n_samples) # chain length does not count adaptation period
      ess_value = compute_ess(chain[(n_samples+1):end,:,:]) # ess does not count adaptation period
      @info "Finished in $(time) seconds."


      while (minimum(ess_value) < MINIMUM_REQUIRED_ESS) && (chain_len < 1000000)
         GC.gc()
         n_samples = ceil(Int, n_samples/1000*max(1,MINIMUM_REQUIRED_ESS*1.5/minimum(ess_value)))*1000
         @info "Redrawing with chain_len=$n_samples (had chain_len=$chain_len)"
         if sampler_name == :NUTS
            chain, time, target_evals, h = sample_hmc(target, 2*n_samples, sampler_seed, n_samples; sampler_specs...);
         else
            error("Sampler not defined")
         end
         adapttime = chain[:current_time][n_samples] - chain[:current_time][1]
         runtime = chain[:current_time][end] - chain[:current_time][n_samples + 1]
         chain_len = copy(n_samples) # chain length does not count adaptation period
         ess_value = compute_ess(chain[(n_samples+1):end,:,:]) # ess does not count adaptation period
         @info "Finished in $(time) seconds."
      end
      @info "Sampling completed!"
   elseif (isnothing(n_rounds) & (sampler_specs.metric_name == :diagonal)) # adapt with time budget
      if sampler_name == :NUTS
         chain, time, target_evals, h = sample_hmc(target, 2*n_samples, sampler_seed, 2*n_samples; sampler_specs...);
      else
         error("Sampler not defined")
      end
      @info "Elapsed adapt time $(time) seconds."
      while time < alotted_adapt_time
         GC.gc()
         # continue adaptation with double chain length
         temp_sampler_specs = (; initstate = vec(Array(chain[end,:,1])), 
            inv_mass_mat = h.metric.M⁻¹, step_size = chain[:step_size][end], sampler_specs...)
         n_samples *= 2
         @info "Continue adapting with specs: $(temp_sampler_specs.step_size)"
         chain, time_temp, target_evals, h = sample_hmc(target, n_samples, sampler_seed, n_samples; temp_sampler_specs...);
         time += time_temp
         @info "Elapsed adapt time $(time) seconds."
      end
      @info "Finished adapting in $(time) seconds."
      adapttime = copy(time)

      # start sampling phase with double budget
      final_sampler_specs = (; initstate = vec(Array(chain[end,:,1])), adapt_mode = :none,
         inv_mass_mat = h.metric.M⁻¹, step_size = chain[:step_size][end], sampler_specs...)
      n_samples *= 2
      chain, time_temp, target_evals, h = sample_hmc(target, n_samples, sampler_seed, 0; final_sampler_specs...);
      time += time_temp

      runtime = chain[:current_time][end] - chain[:current_time][1]
      chain_len = copy(n_samples)
      ess_value = compute_ess(chain) 
      @info "Sampling completed in $(time) !"
   else # round based adaptation (only used for RHMC for now)
      if sampler_name == :NUTS
         chain, time, target_evals, h = sample_hmc(target, 2*n_samples, sampler_seed, 2*n_samples; sampler_specs...);
      else
         error("Sampler not defined")
      end
      @info "Elapsed adapt time $(time) seconds."
      for r in 1:(n_rounds-1)
         GC.gc()
         # continue adaptation with double chain length
         temp_sampler_specs = (; initstate = vec(Array(chain[end,:,1])), 
            step_size = chain[:step_size][end], sampler_specs...)
         n_samples *= 2
         @info "Continue adapting with step size: $(temp_sampler_specs.step_size)"
         chain, time_temp, target_evals, h = sample_hmc(target, n_samples, sampler_seed, n_samples; temp_sampler_specs...);
         time += time_temp
         @info "Elapsed adapt time $(time) seconds."
      end
      @info "Finished adapting in $(time) seconds."
      adapttime = copy(time)

      # start sampling phase with double budget
      final_sampler_specs = (; initstate = vec(Array(chain[end,:,1])), adapt_mode = :none,
         step_size = chain[:step_size][end], sampler_specs...)
      n_samples *= 2
      chain, time_temp, target_evals, h = sample_hmc(target, n_samples, sampler_seed, 0; final_sampler_specs...);
      time += time_temp
      runtime = copy(time_temp)

      if runtime < adapttime
         final_sampler_specs = (; initstate = vec(Array(chain[end,:,1])), adapt_mode = :none,
         step_size = chain[:step_size][end], sampler_specs...)
         n_samples_cont = Int(ceil(adapttime / runtime) - 1) * n_samples

         @info "Not enough run time, sampling $n_samples_cont additional iterations."

         chain_cont, time_cont, target_evals_cont, h = sample_hmc(target, n_samples_cont, sampler_seed, 0; final_sampler_specs...);
         chain = auto_vcat(chain,chain_cont)
         n_samples += n_samples_cont
         time += time_cont
         runtime += time_cont
         target_evals += target_evals_cont
      end

      chain_len = copy(n_samples)
      ess_value = compute_ess(chain) 
      @info "Sampling completed in $(time) !"
   end

   if return_chain
      return chain, runtime, adapttime, target_evals, h
   else
      # return a DataFrame with the results
      ess1_value = compute_ess(chain[(n_samples+1):end,:,:]; degree = 1)
      return DataFrame(
         run_time = runtime, adapt_time = adapttime, chain_length = chain_len,
         dimension = target_specs.dim, target_evals = target_evals,
         min_ess = minimum(ess1_value), median_ess = median(ess1_value), mean_ess = mean(ess1_value), std_ess = std(ess1_value), 
         min_ess2 = minimum(ess_value), median_ess2 = median(ess_value), mean_ess2 = mean(ess_value), std_ess2 = std(ess_value), 
      )
   end
end
