##### DiagRiemannianMetric struct

@auto struct DiagRiemannianMetric <: AbstractRiemannianMetric
    size                # dimension of problem
    a                   # softabs parameter
    ℓπ                  # store target specs
    fp_count            # store number of fixed point steps
    int_fails           # number of times fixed point routine reached max iter
end

Base.size(e::DiagRiemannianMetric) = e.size
Base.size(e::DiagRiemannianMetric, dim::Int) = e.size[dim]


##### Phasepoint for DiagRiemannianMetric

# Note: diagonal of mass matrix is stored in ℓκ.gradient
function AdvancedHMC.phasepoint(
    h::AdvancedHMC.Hamiltonian{<:DiagRiemannianMetric},
    θ::T,
    r::T;
    ℓπ=AdvancedHMC.DualValue([h.ℓπ(θ)],zeros(length(θ),1)), 
    ℓκ=AdvancedHMC.neg_energy(h, r, θ; dual_val = true),
) where {T<:AbstractVecOrMat}
    return AdvancedHMC.PhasePoint(θ, r, ℓπ, ℓκ)
end



##### neg_energy for DiagRiemannianMetric

function AdvancedHMC.neg_energy(
    h::AdvancedHMC.Hamiltonian{<:DiagRiemannianMetric}, r::T, θ::T; dual_val = false
) where {T<:AbstractVecOrMat}
    if isa(h.kinetic, RelativisticKinetic)
        res = rel_kinetic(h.metric.ℓπ, θ, r, h.kinetic.m, h.kinetic.c, h.metric.a; dual_val)
    else
        res = kinetic(h.metric.ℓπ, θ, r, h.metric.a; dual_val)
    end

    res.value[1] = - res.value[1]
    return res
end


##### full leap step

# for implicit midpoint
function full_leap(
    lf::ImplicitMidpoint, 
    h::AdvancedHMC.Hamiltonian{<:DiagRiemannianMetric},
    z::AdvancedHMC.PhasePoint,
    ϵ::Float64,
)
    # @show z

    θ_init, r_init = z.θ, z.r
    target = h.metric.ℓπ
    old_h = AdvancedHMC.neg_energy(z)

    d = length(θ_init)
    # p_init = vcat(θ_init,r_init)

    θ_bar = copy(θ_init)
    r_bar = copy(r_init)

    θ_prev = copy(θ_init)
    r_prev = copy(r_init)

    θ_full = copy(θ_init)
    r_full = copy(r_init)

    rhs_fun = if isa(h.kinetic, RelativisticKinetic)
        function (θ,r)
            θ_bar[:] = (θ + θ_init) / 2
            r_bar[:] = (r + r_init) / 2
            grad_θ, grad_r = zeros(d), zeros(d)
            autodiff(Enzyme.Reverse, rel_hamiltonian, Active, Const(target), Const(target), 
                Duplicated(θ_bar, grad_θ), Duplicated(r_bar, grad_r), Const(h.kinetic.m), Const(h.kinetic.c), Const(h.metric.a))
            θ[:] = θ_init + ϵ * grad_r 
            r[:] = r_init - ϵ * grad_θ
        end
    else
        function (θ,r)
            θ_bar[:] = (θ + θ_init) / 2
            r_bar[:] = (r + r_init) / 2
            grad_θ, grad_r = zeros(d), zeros(d)
            autodiff(Enzyme.Reverse, hamiltonian, Active, Const(target), Const(target), Duplicated(θ_bar, grad_θ), Duplicated(r_bar, grad_r), Const(h.metric.a))
            θ[:] = θ_init + ϵ * grad_r 
            r[:] = r_init - ϵ * grad_θ
        end
    end

    fp_calls, fp_fail = fixedpoint_iter(rhs_fun, θ_full, r_full, θ_prev, r_prev; max_iter = lf.n, rel_tol = lf.δ)
    
    # update phasepoint
    # NOTE:
    z = AdvancedHMC.phasepoint(h, θ_full, r_full)
    h.metric.fp_count[1] += fp_calls
    h.metric.int_fails[1] += fp_fail

    return z 
end


##### derivative functions

function ∂H(    
    h::AdvancedHMC.Hamiltonian{<:DiagRiemannianMetric}, 
    θ::AbstractVecOrMat, 
    r::AbstractVecOrMat
)
    d = length(θ)
    target = h.metric.ℓπ
    grad_θ, grad_r = zeros(d), zeros(d)
    if isa(h.kinetic,RelativisticKinetic)
        autodiff(Enzyme.Reverse, rel_hamiltonian, Active, Const(target), Const(target), 
            Duplicated(θ, grad_θ), Duplicated(r, grad_r), Const(h.kinetic.m), Const(h.kinetic.c), Const(h.metric.a))
    else
        autodiff(Enzyme.Reverse, hamiltonian, Active, Const(target), Const(target), Duplicated(θ, grad_θ), Duplicated(r, grad_r), Const(h.metric.a))
    end
    return grad_θ, grad_r
end

function AdvancedHMC.∂H∂r(
    h::AdvancedHMC.Hamiltonian{<:DiagRiemannianMetric}, 
    θ::AbstractVecOrMat, 
    r::AbstractVecOrMat
)
    res = copy(r)
    temp_vec = get_temp(h.metric.ℓπ, θ)

    for j in eachindex(θ)
        Gii = abs(-second_derivative(h.metric.ℓπ, θ, temp_vec, j, θ[j]))
        if h.metric.a != Inf
            Gii = coth(h.metric.a * Gii) * Gii
        end
        res[j] /= Gii
    end
    return res
end


##### momentum sampling function

function AdvancedHMC.rand_momentum(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::DiagRiemannianMetric{T},
    kinetic,
    θ::AbstractVecOrMat,
) where {T}
    temp_vec = get_temp(metric.ℓπ, θ)
    r = AdvancedHMC.rand_momentum(rng, UnitEuclideanMetric(size(metric)), kinetic, θ)

    for j in eachindex(θ)
        Gii = abs(-second_derivative(metric.ℓπ, θ, temp_vec, j, θ[j]))
        if metric.a != Inf
            Gii = coth(metric.a * Gii) * Gii
        end
        r[j] *= sqrt(Gii)
    end
    
    return r
end

