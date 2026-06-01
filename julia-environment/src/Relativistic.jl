using AdvancedHMC
import AdvancedHMC: ∂H∂r, neg_energy, AbstractKinetic
import Random: AbstractRNG

#################################################################
# Relativistic metric
#################################################################

struct RelativisticKinetic{T} <: AbstractKinetic
    "Mass"
    m::T
    "Speed of light"
    c::T
end

relativistic_mass(kinetic::T, r, r′ = r) where {T} = 
    kinetic.m * sqrt(dot(r, r′) / (kinetic.m ^ 2 * kinetic.c ^ 2) + 1)
relativistic_energy(kinetic::T, r, r′ = r) where {T} = sum(
    kinetic.c ^ 2 * relativistic_mass(kinetic, r, r′)
)


function ∂H∂r(
    h::Hamiltonian{<:UnitEuclideanMetric,<:RelativisticKinetic}, r::AbstractVecOrMat
)
    mass = h.kinetic.m .* sqrt.(r .^ 2 ./ (h.kinetic.m .^ 2 * h.kinetic.c .^ 2) .+ 1)
    return r ./ mass
end
function ∂H∂r(
    h::Hamiltonian{<:DiagEuclideanMetric,<:RelativisticKinetic}, r::AbstractVecOrMat
)
    r = h.metric.sqrtM⁻¹ .* r
    mass = h.kinetic.m .* sqrt.(r .^ 2 ./ (h.kinetic.m .^ 2 * h.kinetic.c .^ 2) .+ 1)
    retval = r ./ mass # red part of (15)
    return h.metric.sqrtM⁻¹ .* retval # (15)
end

function ∂H∂r(h::Hamiltonian{<:DenseRiemannianMetric, <:RelativisticKinetic}, θ::AbstractVecOrMat, r::AbstractVecOrMat)
    M = h.metric.map(h.metric.G(θ))
    Mr = M \ r
    mass = relativistic_mass(h.kinetic, r, Mr)
    return Mr ./ mass
end

function ∂H∂r(
    h::AdvancedHMC.Hamiltonian{<:DiagRiemannianMetric,<:RelativisticKinetic}, 
    θ::AbstractVecOrMat, 
    r::AbstractVecOrMat
)
    res = copy(r)
    temp_vec = get_temp(h.metric.ℓπ, θ)

    for j in eachindex(θ)
        Gii = h.metric.map(-second_derivative(h.metric.ℓπ, θ, temp_vec, j, θ[j]))
        res[j] /= Gii
    end
    mass = h.kinetic.m .* sqrt.(res .* r ./ (h.kinetic.m .^ 2 * h.kinetic.c .^ 2) .+ 1)
    return res ./ mass
end


function neg_energy(
    h::Hamiltonian{<:UnitEuclideanMetric,<:RelativisticKinetic}, r::T, θ::T
) where {T<:AbstractVector}
    return -sum(
        h.kinetic.m .* h.kinetic.c .^ 2 .*
        sqrt.(r .^ 2 ./ (h.kinetic.m .^ 2 .* h.kinetic.c .^ 2) .+ 1),
    )
end


function neg_energy(
    h::Hamiltonian{<:DiagEuclideanMetric,<:RelativisticKinetic}, r::T, θ::T
) where {T<:AbstractVector}
    r = h.metric.sqrtM⁻¹ .* r
    return -sum(
        h.kinetic.m .* h.kinetic.c .^ 2 .*
        sqrt.(r .^ 2 ./ (h.kinetic.m .^ 2 .* h.kinetic.c .^ 2) .+ 1),
    )
end


function neg_energy(
    h::Hamiltonian{<:DenseRiemannianMetric, <:RelativisticKinetic},
    r::T,
    θ::T
) where {T<:AbstractVecOrMat}
    G = h.metric.map(h.metric.G(θ))
    # Need to consider the normalizing term as it is no longer same for different θs
    logZ_partial = 1 / 2 * logdet(G)

    mul!(h.metric._temp, inv(G), r)
    # ldiv!(h.metric._temp, cholesky(G), r) # https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.ldiv!
    return -relativistic_energy(h.kinetic, r, h.metric._temp) - logZ_partial 
end



using AdaptiveRejectionSampling: RejectionSampler, run_sampler!
import ForwardDiff: derivative 

# remove info popup for RejectionSampler
function RejectionSampler(
        f::Function,
        support::Tuple{Float64,Float64},
        δ::Float64=0.5;
        search_range::Tuple{Float64,Float64}=(-10.0, 10.0),
        min_slope::Float64=1e-6,
        max_slope::Float64=10.0,
        logdensity::Bool=false,
        kwargs...
    )
        if logdensity
            logf = f
        else
            logf = (x -> log(f(x)))
        end
        grad(x) = derivative(logf, x)
        grid_lims = max(search_range[1], support[1]), min(search_range[2], support[2])
        grid = grid_lims[1]:δ:grid_lims[2]
        grads = grad.(grid)
        i1 = findfirst(min_slope .< grads .< max_slope)
        i2 = findlast(min_slope .< - grads .< max_slope)
        @assert (i1 !== nothing) && (i2 !== nothing) "couldn't find initial points, please provide them or change `search_range`"
        @assert i1 < i2 "function is not logconcave, first index with grad>0 must be smaller than first index with grad<0"
        x1, x2 = grid[i1], grid[i2]
        RejectionSampler(f, support, (x1, x2); logdensity=logdensity, kwargs...)
    end

function AdvancedHMC.rand_momentum(
    rng::AbstractRNG, metric::UnitEuclideanMetric{T}, kinetic::RelativisticKinetic{T}, θ::AbstractVecOrMat
) where {T}
    h_temp = Hamiltonian(metric, kinetic, identity, identity)
    densityfunc = x -> exp(neg_energy(h_temp, [x], [x]))
    sampler = RejectionSampler(densityfunc, (-Inf, Inf); max_segments=5)
    sz = size(metric)
    r = run_sampler!(rng, sampler, prod(sz))
    r = reshape(r, sz)
    return r
end

function AdvancedHMC.rand_momentum(
    rng::AbstractRNG, metric::DiagEuclideanMetric{T}, kinetic::RelativisticKinetic{T}, θ::AbstractVecOrMat,
) where {T}
    r = AdvancedHMC.rand_momentum(rng, UnitEuclideanMetric(size(metric)), kinetic, θ)
    # p' = A p where A = sqrtM
    r ./= metric.sqrtM⁻¹
    return r
end

