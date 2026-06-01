using Random

import AdvancedHMC: ∂H∂θ, ∂H∂r, DualValue, PhasePoint, phasepoint, step
using AdvancedHMC: TYPEDEF, TYPEDFIELDS, AbstractScalarOrVec, AbstractLeapfrog, step_size, AbstractMetric
import AdvancedHMC: phasepoint, neg_energy, ∂H∂θ, ∂H∂r
using LinearAlgebra: logabsdet, tr, eigen, cholesky, Symmetric


#################################################################
# Riemannian metric
#################################################################

##### Utility functions for AdvancedHMC

# copy functions
Base.copy(v::AdvancedHMC.DualValue{}) = AdvancedHMC.DualValue(copy(v.value),copy(v.gradient))
Base.copy(z::AdvancedHMC.PhasePoint{}) = AdvancedHMC.PhasePoint(copy(z.θ), copy(z.r), copy(z.ℓπ), copy(z.ℓκ))

# neg_energy for phasepoint
AdvancedHMC.neg_energy(z::AdvancedHMC.PhasePoint{}) = z.ℓπ.value[1] + z.ℓκ.value[1]


##### DenseRiemannianMetric struct

abstract type AbstractRiemannianMetric <: AbstractMetric end

@auto struct DenseRiemannianMetric <: AbstractRiemannianMetric
    size
    G
    ∂G∂θ
    map
    _temp
    fp_count
    int_fails
end

# TODO Make dense mass matrix support matrix-mode parallel
function DenseRiemannianMetric(size, G, ∂G∂θ, map=IdentityMap())
    _temp = Vector{Float64}(undef, size[1])
    return DenseRiemannianMetric(size, G, ∂G∂θ, map, _temp,[0],[0])
end

Base.size(e::DenseRiemannianMetric) = e.size
Base.size(e::DenseRiemannianMetric, dim::Int) = e.size[dim]
Base.show(io::IO, dem::DenseRiemannianMetric) = print(io, "DenseRiemannianMetric(...)")

##### Functions in these files are used below
include("Integrators.jl")
include("Diagonal_Riemannian.jl")
include("Relativistic.jl")

##### Phasepoint for DenseRiemannianMetric

function AdvancedHMC.phasepoint(
    h::AdvancedHMC.Hamiltonian{<:DenseRiemannianMetric},
    θ::T,
    r::T;
    ℓπ = ∂H∂θ(h, θ, r),
    ℓκ = DualValue(neg_energy(h, r, θ), ∂H∂r(h, θ, r)),
) where {T<:AbstractVecOrMat}
    return AdvancedHMC.PhasePoint(θ, r, ℓπ, ℓκ)
end


##### neg_energy for DenseRiemannianMetric

#! Eq (13) of Girolami & Calderhead (2011)
function neg_energy(
    h::Hamiltonian{<:DenseRiemannianMetric}, r::T, θ::T
) where {T<:AbstractVecOrMat}
    G = h.metric.map(h.metric.G(θ))
    D = size(G, 1)
    # Need to consider the normalizing term as it is no longer same for different θs
    logZ = 1 / 2 * (D * log(2π) + logdet(G)) # it will be user's responsibility to make sure G is SPD and logdet(G) is defined
    mul!(h.metric._temp, inv(G), r)
    return -logZ - dot(r, h.metric._temp) / 2
end


##### full leap step

# for implicit midpoint
function full_leap(
    lf::ImplicitMidpoint, 
    h::AdvancedHMC.Hamiltonian{<:DenseRiemannianMetric},
    z::AdvancedHMC.PhasePoint,
    ϵ::Float64,
)
    # @show z

    θ_init, r_init = z.θ, z.r

    d = length(θ_init)
    # p_init = vcat(θ_init,r_init)

    θ_bar = copy(θ_init)
    r_bar = copy(r_init)

    θ_prev = copy(θ_init)
    r_prev = copy(r_init)

    θ_full = copy(θ_init)
    r_full = copy(r_init)

    rhs_fun = function (θ,r)
        θ_bar[:] = (θ + θ_init) / 2
        r_bar[:] = (r + r_init) / 2
        θ[:] = θ_init + ϵ * ∂H∂r(h, θ_bar, r_bar)
        r[:] = r_init - ϵ * ∂H∂θ(h, θ_bar, r_bar).gradient
    end


    fp_calls, fp_fail = fixedpoint_iter(rhs_fun, θ_full, r_full, θ_prev, r_prev; max_iter = lf.n, rel_tol = lf.δ)
    
    # update phasepoint
    z = AdvancedHMC.phasepoint(h, θ_full, r_full)
    h.metric.fp_count[1] += fp_calls
    h.metric.int_fails[1] += fp_fail

    return z 
end

# for generalized leapfrog
function full_leap(
    lf::GeneralizedLeapfrog, 
    h::AdvancedHMC.Hamiltonian{<:DenseRiemannianMetric},
    z::AdvancedHMC.PhasePoint,
    ϵ::Float64,
)
    θ_init, r_init = z.θ, z.r
    # Tempering
    #r = temper(lf, r, (i=i, is_half=true), n_steps)
    #! Eq (16) of Girolami & Calderhead (2011)
    r_half = copy(r_init)
    local cache
    for j = 1:lf.n
        # Reuse cache for the first iteration
        if j == 1
            (; value, gradient) = z.ℓπ
        elseif j == 2 # cache intermediate values that depends on θ only (which are unchanged)
            retval, cache = ∂H∂θ_cache(h, θ_init, r_half; return_cache = true)
            (; value, gradient) = retval
        else # reuse cache
            (; value, gradient) = ∂H∂θ_cache(h, θ_init, r_half; cache = cache)
        end
        r_half = r_init - ϵ / 2 * gradient
    end
    
    #! Eq (17) of Girolami & Calderhead (2011)
    θ_full = copy(θ_init)
    term_1 = ∂H∂r(h, θ_init, r_half) # unchanged across the loop
    for j = 1:lf.n
        θ_full = θ_init + ϵ / 2 * (term_1 + ∂H∂r(h, θ_full, r_half))
    end
    
    #! Eq (18) of Girolami & Calderhead (2011)
    (; value, gradient) = ∂H∂θ(h, θ_full, r_half)
    r_full = r_half - ϵ / 2 * gradient
    
    # Create a new phase point by caching the logdensity and gradient
    z = phasepoint(h, θ_full, r_full; ℓπ = DualValue(value, gradient))

    h.metric.fp_count[1] += lf.n

    return z 
end



##### Derivative functions

# Fallback to ignore return_cache & cache kwargs for other ∂H∂θ
function ∂H∂θ_cache(h, θ::AbstractArray{T}, r::AbstractArray{T}; return_cache=false, cache=nothing) where {T}
    dv = ∂H∂θ(h, θ, r)
    return return_cache ? (dv, nothing) : dv
end

### Hessian map utilities

abstract type AbstractHessianMap end

struct IdentityMap <: AbstractHessianMap end

(::IdentityMap)(x) = x

struct SoftAbsMap{T} <: AbstractHessianMap
    α::T
end

# TODO Register softabs with ReverseDiff
#! The definition of SoftAbs from Page 3 of Betancourt (2012)
function softabs(X, α=20.0)
    F = eigen(X) # ReverseDiff cannot diff through `eigen`
    Q = hcat(F.vectors)
    λ = F.values
    softabsλ = λ .* coth.(α * λ)
    return Q * diagm(softabsλ) * Q', Q, λ, softabsλ
end

(map::SoftAbsMap)(x) = softabs(x, map.α)[1]

# Ref: https://www.wolframalpha.com/input?i=derivative+of+x+*+coth%28a+*+x%29
#! Based on middle of the right column of Page 3 of Betancourt (2012) "Note that whenλi=λj, such as for the diagonal elementsor degenerate eigenvalues, this becomes the derivative"
dsoftabsdλ(α, λ) = coth(α * λ) + λ * α * -csch(λ * α)^2

#! J as defined in middle of the right column of Page 3 of Betancourt (2012)
function make_J(λ::AbstractVector{T}, α::T) where {T<:AbstractFloat}
    d = length(λ)
    J = Matrix{T}(undef, d, d)
    for i in 1:d, j in 1:d
        J[i, j] = if (λ[i] == λ[j])
            dsoftabsdλ(α, λ[i])
        else
            ((λ[i] * coth(α * λ[i]) - λ[j] * coth(α * λ[j])) / (λ[i] - λ[j]))
        end
    end
    return J
end


### Derivatives of H

# ∂H∂θ for identity map
# QUES L31 of hamiltonian.jl now reads a bit weird (semantically)
function ∂H∂θ(
    h::Hamiltonian{<:DenseRiemannianMetric{<:Any,<:Any,<:Any,<:IdentityMap}, <:GaussianKinetic},
    θ::AbstractVecOrMat,
    r::AbstractVecOrMat,
)
    ℓπ, ∂ℓπ∂θ = h.∂ℓπ∂θ(θ)
    G = h.metric.map(h.metric.G(θ))
    invG = inv(G)
    ∂G∂θ = h.metric.∂G∂θ(θ)
    d = length(∂ℓπ∂θ)

    isdiag = ∂G∂θ isa AbstractMatrix
    if isdiag
        g = -mapreduce(vcat, 1:d) do i
            ∂G∂θᵢ = ∂G∂θ[:, i]
            diag_invG = diag(invG)
            ∂ℓπ∂θ[i] - 1 / 2 * sum(diag_invG .* ∂G∂θᵢ) + 1 / 2 * r' * (diag_invG .* ∂G∂θᵢ .* diag_invG .* r)
        end
    else
        g = -mapreduce(vcat, 1:d) do i
            ∂G∂θᵢ = ∂G∂θ[:, :, i]
            ∂ℓπ∂θ[i] - 1 / 2 * tr(invG * ∂G∂θᵢ) + 1 / 2 * r' * invG * ∂G∂θᵢ * invG * r
        end
    end

    return DualValue(ℓπ, g)
end

# ∂H∂θ for identity map + relativistic kinetic
function ∂H∂θ(
    h::Hamiltonian{<:DenseRiemannianMetric{<:Any,<:Any,<:Any,<:IdentityMap}, <:RelativisticKinetic},
    θ::AbstractVecOrMat,
    r::AbstractVecOrMat,
)
    ℓπ, ∂ℓπ∂θ = h.∂ℓπ∂θ(θ)
    G = h.metric.map(h.metric.G(θ))
    invG = inv(G)
    ∂G∂θ = h.metric.∂G∂θ(θ)
    d = length(∂ℓπ∂θ)

    mass = relativistic_mass(h.kinetic, r, G \ r)
    
    isdiag = ∂G∂θ isa AbstractMatrix
    if isdiag
        g = -mapreduce(vcat, 1:d) do i
            ∂G∂θᵢ = ∂G∂θ[:, i]
            diag_invG = diag(invG)
            ∂ℓπ∂θ[i] - 1 / 2 * sum(diag_invG .* ∂G∂θᵢ) + 1 / (2 * mass) * r' * (diag_invG .* ∂G∂θᵢ .* diag_invG .* r)
        end
    else
        g = -mapreduce(vcat, 1:d) do i
            ∂G∂θᵢ = ∂G∂θ[:, :, i]
            ∂ℓπ∂θ[i] - 1 / 2 * tr(invG * ∂G∂θᵢ) + 1 / (2 * mass) * r' * invG * ∂G∂θᵢ * invG * r
        end
    end

    return DualValue(ℓπ, g)
end


# ∂H∂θ for softabs map
function ∂H∂θ(
    h::Hamiltonian{<:DenseRiemannianMetric{<:Any,<:Any,<:Any,<:SoftAbsMap}, <:GaussianKinetic},
    θ::AbstractVecOrMat,
    r::AbstractVecOrMat,
)
    return ∂H∂θ_cache(h, θ, r)
end
function ∂H∂θ_cache(
    h::Hamiltonian{<:DenseRiemannianMetric{<:Any,<:Any,<:Any,<:SoftAbsMap}, <:GaussianKinetic},
    θ::AbstractVecOrMat,
    r::AbstractVecOrMat;
    return_cache=false,
    cache=nothing,
)
    # Terms that only dependent on θ can be cached in θ-unchanged loops
    if isnothing(cache)
        ℓπ, ∂ℓπ∂θ = h.∂ℓπ∂θ(θ)
        H = h.metric.G(θ)
        ∂H∂θ = h.metric.∂G∂θ(θ)

        G, Q, λ, softabsλ = softabs(H, h.metric.map.α)

        R = Diagonal(1 ./ softabsλ)

        # softabsΛ = Diagonal(softabsλ)
        # M = inv(softabsΛ) * Q' * r
        # M = R * Q' * r # equiv to above but avoid inv

        J = make_J(λ, h.metric.map.α)

        #! Based on the two equations from the right column of Page 3 of Betancourt (2012)
        term_1_cached = Q * (R .* J) * Q'
    else
        (; ℓπ, ∂ℓπ∂θ, ∂H∂θ, Q, softabsλ, J, term_1_cached) = cache
    end
    d = length(∂ℓπ∂θ)
    D = Diagonal((Q' * r) ./ softabsλ)
    term_2_cached = Q * D * J * D * Q'

    isdiag = ∂H∂θ isa AbstractMatrix
    g = isdiag ?
        -(∂ℓπ∂θ - 1 / 2 * diag(term_1_cached * ∂H∂θ) + 1 / 2 * diag(term_2_cached * ∂H∂θ)) :
        -mapreduce(vcat, 1:d) do i
            ∂H∂θᵢ = ∂H∂θ[:, :, i]
            # ∂ℓπ∂θ[i] - 1 / 2 * tr(term_1_cached * ∂H∂θᵢ) + 1 / 2 * M' * (J .* (Q' * ∂H∂θᵢ * Q)) * M # (v1)
            # NOTE Some further optimization can be done here: cache the 1st product all together
            ∂ℓπ∂θ[i] - 1 / 2 * tr(term_1_cached * ∂H∂θᵢ) + 1 / 2 * tr(term_2_cached * ∂H∂θᵢ) # (v2) cache friendly
        end

    dv = DualValue(ℓπ, g)
    return return_cache ? (dv, (; ℓπ, ∂ℓπ∂θ, ∂H∂θ, Q, softabsλ, J, term_1_cached)) : dv
end

# ∂H∂θ for softabs map + relativistic kinetic
∂H∂θ(
    h::Hamiltonian{<:DenseRiemannianMetric{<:Any,<:Any,<:Any,<:SoftAbsMap}, <:RelativisticKinetic},
    θ::AbstractVecOrMat,
    r::AbstractVecOrMat,
) = ∂H∂θ_cache(h, θ, r)
function ∂H∂θ_cache(
    h::Hamiltonian{<:DenseRiemannianMetric{<:Any,<:Any,<:Any,<:SoftAbsMap}, <:RelativisticKinetic},
    θ::AbstractVecOrMat,
    r::AbstractVecOrMat;
    return_cache = false,
    cache = nothing,
)
    # Terms that only dependent on θ can be cached in θ-unchanged loops
    if isnothing(cache)
        ℓπ, ∂ℓπ∂θ = h.∂ℓπ∂θ(θ)
        H = h.metric.G(θ)
        ∂H∂θ = h.metric.∂G∂θ(θ)

        G, Q, λ, softabsλ = softabs(H, h.metric.map.α)

        R = Diagonal(1 ./ softabsλ)

        # softabsΛ = Diagonal(softabsλ)
        # M = inv(softabsΛ) * Q' * r
        # M = R * Q' * r # equiv to above but avoid inv

        J = make_J(λ, h.metric.map.α)

        #! Based on the two equations from the right column of Page 3 of Betancourt (2012)
        term_1_cached = Q * (R .* J) * Q'

        M = G
    else
        (; ℓπ, ∂ℓπ∂θ, ∂H∂θ, Q, softabsλ, J, term_1_cached, M) = cache
    end
    d = length(∂ℓπ∂θ)
    D = Diagonal((Q' * r) ./ softabsλ)
    term_2 = Q * D * J * D * Q'

    #! (18) of Overleaf note
    mass = relativistic_mass(h.kinetic, r, M \ r)

    isdiag = ∂H∂θ isa AbstractMatrix
    g = isdiag ?
        -(∂ℓπ∂θ - 1 / 2 * diag(term_1_cached * ∂H∂θ) + 1 / 2 * (1 / mass) * diag(term_2 * ∂H∂θ)) :
        -mapreduce(vcat, 1:d) do i
            ∂H∂θᵢ = ∂H∂θ[:,:,i]
            ∂ℓπ∂θ[i] - 1 / 2 * tr(term_1_cached * ∂H∂θᵢ) + 1 / 2 * (1 / mass) * tr(term_2 * ∂H∂θᵢ) # (v2) cache friendly
        end

    dv = DualValue(ℓπ, g)
    return return_cache ? (dv, (; ℓπ, ∂ℓπ∂θ, ∂H∂θ, Q, softabsλ, J, term_1_cached, M)) : dv
end

# ∂H∂r
#! Eq (14) of Girolami & Calderhead (2011)
function ∂H∂r(
    h::Hamiltonian{<:DenseRiemannianMetric}, θ::AbstractVecOrMat, r::AbstractVecOrMat
)
    H = h.metric.G(θ)
    G = h.metric.map(H)
    return G \ r
end



##### momentum sampling function

function AdvancedHMC.rand_momentum(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::DenseRiemannianMetric,
    kinetic,
    θ::AbstractVecOrMat,
)
    r = AdvancedHMC.rand_momentum(rng, UnitEuclideanMetric(size(metric)), kinetic, θ)
    # @show r
    G⁻¹ = inv(metric.map(metric.G(θ)))
    chol = cholesky(Symmetric(G⁻¹))
    # @show chol.U
    # @show metric.map
    # @show G⁻¹
    ldiv!(chol.U, r)
    # @show r
    return r
end


##### Trajectory termination for state dependent HMC samplers

function AdvancedHMC.TurnStatistic(
    ::Union{AdvancedHMC.GeneralisedNoUTurn,AdvancedHMC.StrictGeneralisedNoUTurn}, 
    z::AdvancedHMC.PhasePoint,
    h::AdvancedHMC.Hamiltonian{<:AbstractRiemannianMetric}
)
    return AdvancedHMC.TurnStatistic(z.r)
    # return AdvancedHMC.TurnStatistic(AdvancedHMC.∂H∂r(h, z.θ, z.r))
end


function AdvancedHMC.isterminated(::AdvancedHMC.GeneralisedNoUTurn, h::AdvancedHMC.Hamiltonian{<:AbstractRiemannianMetric}, t::AdvancedHMC.BinaryTree)
    rho = t.ts.rho
    s = AdvancedHMC.generalised_uturn_criterion(rho, AdvancedHMC.∂H∂r(h, t.zleft.θ, t.zleft.r), AdvancedHMC.∂H∂r(h, t.zright.θ, t.zright.r))
    # s = AdvancedHMC.generalised_uturn_criterion(rho, t.zleft.r, t.zright.r)
    return AdvancedHMC.Termination(s, false)
end


function AdvancedHMC.build_tree(
    rng::AbstractRNG,
    nt::AdvancedHMC.Trajectory{TS,I,TC},
    h::AdvancedHMC.Hamiltonian{<:AbstractRiemannianMetric},
    z::AdvancedHMC.PhasePoint,
    sampler::AdvancedHMC.AbstractTrajectorySampler,
    v::Int,
    j::Int,
    H0::AbstractFloat,
) where {
    TS<:AdvancedHMC.AbstractTrajectorySampler,I<:AdvancedHMC.AbstractIntegrator,TC<:AdvancedHMC.DynamicTerminationCriterion
}
    if j == 0
        # Base case - take one leapfrog step in the direction v.
        z′ = AdvancedHMC.step(nt.integrator, h, z, v)
        H′ = AdvancedHMC.energy(z′)
        ΔH = H′ - H0
        α′ = exp(min(0, -ΔH))
        sampler′ = TS(sampler, H0, z′)
        return AdvancedHMC.BinaryTree(z′, z′, AdvancedHMC.TurnStatistic(nt.termination_criterion, z′, h), α′, 1, ΔH),
        sampler′,
        AdvancedHMC.Termination(sampler′, nt, H0, H′)
    else
        # Recursion - build the left and right subtrees.
        tree′, sampler′, termination′ = AdvancedHMC.build_tree(rng, nt, h, z, sampler, v, j - 1, H0)
        # Expand tree if not terminated
        if !AdvancedHMC.isterminated(termination′)
            # Expand left
            if v == -1
                tree′′, sampler′′, termination′′ = AdvancedHMC.build_tree(
                    rng, nt, h, tree′.zleft, sampler, v, j - 1, H0
                ) # left tree
                treeleft, treeright = tree′′, tree′
                # Expand right
            else
                tree′′, sampler′′, termination′′ = AdvancedHMC.build_tree(
                    rng, nt, h, tree′.zright, sampler, v, j - 1, H0
                ) # right tree
                treeleft, treeright = tree′, tree′′
            end
            tree′ = AdvancedHMC.combine(treeleft, treeright)
            sampler′ = AdvancedHMC.combine(rng, sampler′, sampler′′)
            termination′ =
                termination′ *
                termination′′ *
                AdvancedHMC.isterminated(nt.termination_criterion, h, tree′, treeleft, treeright)
        end
        return tree′, sampler′, termination′
    end
end

function AdvancedHMC.transition(
    rng::AbstractRNG, τ::AdvancedHMC.Trajectory{TS,I,TC}, h::AdvancedHMC.Hamiltonian{<:AbstractRiemannianMetric}, z0::AdvancedHMC.PhasePoint
) where {
    TS<:AdvancedHMC.AbstractTrajectorySampler,I<:AdvancedHMC.AbstractIntegrator,TC<:AdvancedHMC.DynamicTerminationCriterion
}
    H0 = AdvancedHMC.energy(z0)
    tree = AdvancedHMC.BinaryTree(
        z0, z0, AdvancedHMC.TurnStatistic(τ.termination_criterion, z0, h), zero(H0), zero(Int), zero(H0)
    )
    sampler = TS(rng, z0)
    termination = AdvancedHMC.Termination(false, false)
    zcand = z0

    j = 0
    while !AdvancedHMC.isterminated(termination) && j < τ.termination_criterion.max_depth
        # Sample a direction; `-1` means left and `1` means right
        vleft = rand(rng, Bool)
        if vleft
            # Create a tree with depth `j` on the left
            tree′, sampler′, termination′ = AdvancedHMC.build_tree(
                rng, τ, h, tree.zleft, sampler, -1, j, H0
            )
            treeleft, treeright = tree′, tree
        else
            # Create a tree with depth `j` on the right
            tree′, sampler′, termination′ = AdvancedHMC.build_tree(
                rng, τ, h, tree.zright, sampler, 1, j, H0
            )
            treeleft, treeright = tree, tree′
        end
        # Perform a MH step and increse depth if not terminated
        if !AdvancedHMC.isterminated(termination′)
            j = j + 1   # increment tree depth
            if AdvancedHMC.mh_accept(rng, sampler, sampler′)
                zcand = sampler′.zcand
            end
        end
        # Combine the proposed tree and the current tree (no matter terminated or not)
        tree = AdvancedHMC.combine(treeleft, treeright)
        # Update sampler
        sampler = AdvancedHMC.combine(zcand, sampler, sampler′)
        # update termination
        termination =
            termination *
            termination′ *
            AdvancedHMC.isterminated(τ.termination_criterion, h, tree, treeleft, treeright)
    end

    H = AdvancedHMC.energy(zcand)
    tstat = merge(
        (
            n_steps=tree.nα,
            is_accept=true,
            acceptance_rate=tree.sum_α / tree.nα,
            log_density=zcand.ℓπ.value,
            hamiltonian_energy=H,
            hamiltonian_energy_error=H - H0,
            max_hamiltonian_energy_error=tree.ΔH_max,
            tree_depth=j,
            numerical_error=termination.numerical,
            gradient_calls=h.metric.fp_count[1],
            integrator_fails=h.metric.int_fails[1],
        ),
        AdvancedHMC.stat(τ.integrator),
    )

    return AdvancedHMC.Transition(zcand, tstat)
end

function AdvancedHMC.transition(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    τ::AdvancedHMC.Trajectory{TS,I,TC},
    h::AdvancedHMC.Hamiltonian{<:AbstractRiemannianMetric},
    z::AdvancedHMC.PhasePoint,
) where {TS<:AdvancedHMC.AbstractTrajectorySampler,I,TC<:AdvancedHMC.StaticTerminationCriterion}
    H0 = AdvancedHMC.energy(z)

    z′, is_accept, α = AdvancedHMC.sample_phasepoint(rng, τ, h, z)
    # Do the actual accept / reject
    z = AdvancedHMC.accept_phasepoint!(z, z′, is_accept)    # NOTE: this function changes `z′` in place in matrix-parallel mode
    # Reverse momentum variable to preserve reversibility
    z = AdvancedHMC.PhasePoint(z.θ, -z.r, z.ℓπ, z.ℓκ)
    # Get cached hamiltonian energy 
    H, H′ = AdvancedHMC.energy(z), AdvancedHMC.energy(z′)
    tstat = merge(
        (
            n_steps=AdvancedHMC.nsteps(τ),
            is_accept=is_accept,
            acceptance_rate=α,
            log_density=z.ℓπ.value,
            hamiltonian_energy=H,
            hamiltonian_energy_error=H - H0,
            # check numerical error in proposed phase point. 
            numerical_error=!all(isfinite, H′),
            gradient_calls=h.metric.fp_count[1],
            integrator_fails=h.metric.int_fails[1],
        ),
        AdvancedHMC.stat(τ.integrator),
    )
    return AdvancedHMC.Transition(z, tstat)
end
