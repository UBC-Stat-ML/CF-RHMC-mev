###########################################################
# Modify AdvancedHMC functions to return more components
###########################################################

function AdvancedHMC.sample(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    h::AdvancedHMC.Hamiltonian,
    κ::AdvancedHMC.HMCKernel,
    θ::T,
    n_samples::Int,
    adaptor::AdvancedHMC.AbstractAdaptor=AdvancedHMC.NoAdaptation(),
    n_adapts::Int=min(div(n_samples, 10), 1_000);
    drop_warmup=false,
    verbose::Bool=true,
    progress::Bool=false,
    (pm_next!)::Function=AdvancedHMC.pm_next!,
) where {T<:AbstractVecOrMat{<:AbstractFloat}}
    @assert !(drop_warmup && (adaptor isa AdvancedHMC.Adaptation.NoAdaptation)) "Cannot drop warmup samples if there is no adaptation phase."
    # Prepare containers to store sampling results
    n_keep = n_samples - (drop_warmup ? n_adapts : 0)
    θs, stats = Vector{T}(undef, n_keep), Vector{NamedTuple}(undef, n_keep)
    num_divergent_transitions = 0
    num_divergent_transitions_during_adaption = 0
    # Initial sampling
    h, t = AdvancedHMC.sample_init(rng, h, θ)
    # Progress meter
    pm = progress ? AdvancedHMC.ProgressMeter.Progress(n_samples; desc="Sampling", barlen=31) : nothing
    time = @elapsed for i in 1:n_samples
        # Make a transition
        t = AdvancedHMC.transition(rng, h, κ, t.z)
        # Adapt h and κ; what mutable is the adaptor
        tstat = AdvancedHMC.stat(t)
        h, κ, isadapted = AdvancedHMC.adapt!(h, κ, adaptor, i, n_adapts, t.z.θ, tstat.acceptance_rate)
        if isadapted
            num_divergent_transitions_during_adaption += tstat.numerical_error
        else
            num_divergent_transitions += tstat.numerical_error
        end
        tstat = AdvancedHMC.merge(tstat, (is_adapt=isadapted,))
        # Update progress meter
        if pm !== nothing
            percentage_divergent_transitions =
                num_divergent_transitions / max(i - n_adapts, 1)
            percentage_divergent_transitions_during_adaption =
                num_divergent_transitions_during_adaption / min(i, n_adapts)
            if percentage_divergent_transitions > 0.25
                @warn "The level of numerical errors is high. Please check the model carefully." maxlog =
                    3
            end
            # Do include current iteration and mass matrix
            pm_next!(
                pm,
                (
                    iterations=i,
                    ratio_divergent_transitions=round(
                        percentage_divergent_transitions; digits=2
                    ),
                    ratio_divergent_transitions_during_adaption=round(
                        percentage_divergent_transitions_during_adaption; digits=2
                    ),
                    tstat...,
                    mass_matrix=h.metric,
                ),
            )
            # Report finish of adapation
        elseif verbose && isadapted && i == n_adapts
            @info "Finished $n_adapts adapation steps" adaptor κ.τ.integrator h.metric
        end
        # Store sample
        if !drop_warmup || i > n_adapts
            j = i - drop_warmup * n_adapts
            θs[j], stats[j] = t.z.θ, tstat
        end
    end
    # Report end of sampling
    if verbose
        EBFMI_est = AdvancedHMC.EBFMI(map(s -> s.hamiltonian_energy, stats))
        average_acceptance_rate = mean(map(s -> s.acceptance_rate, stats))
        if θ isa AbstractVector
            n_chains = 1
        else
            n_chains = size(θ, 2)
            # Make sure that arrays are on CPU before printing.
            EBFMI_est = convert(Vector{eltype(EBFMI_est)}, EBFMI_est)
            average_acceptance_rate = convert(
                Vector{eltype(average_acceptance_rate)}, average_acceptance_rate
            )
            EBFMI_est = "[" * join(EBFMI_est, ", ") * "]"
            average_acceptance_rate = "[" * join(average_acceptance_rate, ", ") * "]"
        end
        @info "Finished $n_samples sampling steps for $n_chains chains in $time (s)" h κ EBFMI_est average_acceptance_rate
    end
    return θs, stats, h
end