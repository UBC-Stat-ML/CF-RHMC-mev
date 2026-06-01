#################################################################
# Functions to make sure Hamiltonian generation using custom
# kinetic goes smoothly
#################################################################

## Without explicit AD backend
function relHamiltonian(metric::AdvancedHMC.AbstractMetric, kinetic::AdvancedHMC.AbstractKinetic, ℓ::AbstractMCMC.LogDensityModel; kwargs...)
    return AdvancedHMC.Hamiltonian(metric, kinetic, ℓ.logdensity; kwargs...)
end
function AdvancedHMC.Hamiltonian(metric::AdvancedHMC.AbstractMetric, kinetic::AdvancedHMC.AbstractKinetic, ℓ; kwargs...)
    cap = LogDensityProblems.capabilities(ℓ)
    if cap === nothing
        throw(
            ArgumentError(
                "The log density function does not support the LogDensityProblems.jl interface",
            ),
        )
    end
    # Check if we're capable of computing gradients.
    ℓπ = if cap === LogDensityProblems.LogDensityOrder{0}()
        # In this case ℓ does not support evaluation of the gradient of the log density function
        # We use ForwardDiff to compute the gradient
        LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), ℓ; kwargs...)
    else
        # In this case ℓ already supports evaluation of the gradient of the log density function
        ℓ
    end
    return Hamiltonian(
        metric,
        kinetic,
        Base.Fix1(LogDensityProblems.logdensity, ℓπ),
        Base.Fix1(LogDensityProblems.logdensity_and_gradient, ℓπ),
    )
end

## With explicit AD specification
function relHamiltonian(
    metric::AdvancedHMC.AbstractMetric, kinetic::AdvancedHMC.AbstractKinetic, ℓπ::AbstractMCMC.LogDensityModel, kind::Union{Symbol,Val,Module}; kwargs...
)
    return AdvancedHMC.Hamiltonian(metric, kinetic, ℓπ.logdensity, kind; kwargs...)
end
function relHamiltonian(metric::AdvancedHMC.AbstractMetric, kinetic::AdvancedHMC.AbstractKinetic, ℓπ, kind; kwargs...)
    if LogDensityProblems.capabilities(ℓπ) === nothing
        throw(
            ArgumentError(
                "The log density function does not support the LogDensityProblems.jl interface",
            ),
        )
    end
    ℓ = LogDensityProblemsAD.ADgradient(
        kind, ℓπ; kwargs...
    )
    return AdvancedHMC.Hamiltonian(metric, kinetic, ℓ)
end
