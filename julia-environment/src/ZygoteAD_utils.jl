#################################################################
# Metric functions used in AdvancedHMC
#################################################################

gen_hess_rev(func, x::AbstractVector) = MCMCLogDensityProblems.gen_hess(func, x)

function gen_hess_hybrid_diag(func, x::AbstractVector)
    grad = MCMCLogDensityProblems.gen_grad(func, x)
    function hess(x::AbstractVector)
        v, g = grad(x)
        h, = diaghessian(func, x)
        return v, g, Diagonal(h)
    end
    return hess
end

function gen_hess_fwd(func, x::AbstractVector)
    function hess(x::AbstractVector)
        return nothing, nothing, ForwardDiff.hessian(func, x)
    end
    return hess
end

function gen_hess_fwd_diag(func, x::AbstractVector)
    function hess(x::AbstractVector)
        h, = diaghessian(func, x)
        return nothing, nothing, Diagonal(h)
    end
    return hess
end

function gen_∂G∂θ_fwd(Vfunc, x; f = identity, gen_hess = gen_hess_fwd)
    _Hfunc = gen_hess(Vfunc, x)
    Hfunc = x -> f(_Hfunc(x)[3])
    # QUES What's the best output format of this function?
    cfg = ForwardDiff.JacobianConfig(Hfunc, x)
    d = length(x)
    out = zeros(eltype(x), d^2, d)
    # default output shape [∂H∂x₁; ∂H∂x₂; ...]
    return x -> ForwardDiff.jacobian!(out, Hfunc, x, cfg)
end

function gen_∂G∂θ_fwd_diag(Vfunc, x; f = identity, gen_hess = gen_hess_fwd)
    _Hfunc = gen_hess(Vfunc, x)
    Hfunc = x -> f(_Hfunc(x)[3]).diag
    # QUES What's the best output format of this function?
    cfg = ForwardDiff.JacobianConfig(Hfunc, x)
    T = eltype(x)
    d = length(x)
    out = zeros(T, d, d)
    # TODO check it needs a transpose
    # default output shape w/ d = 3
    # d_x1_x1_x1, d_x1_x1_x2, d_x1_x1_x3
    # d_x2_x2_x1, d_x2_x2_x2, d_x2_x2_x3
    # d_x3_x3_x1, d_x3_x3_x2, d_x3_x3_x3
    return x -> ForwardDiff.jacobian!(out, Hfunc, x, cfg)
end

function reshape_∂G∂θ(H)
    d = size(H, 2)
    return cat((H[(i-1)*d+1:i*d, :] for i = 1:d)...; dims = 3)
end

_unit_matrix(H::Diagonal) = diagm(ones(size(H, 1)))
_unit_matrix(H::AbstractMatrix) = Diagonal(ones(size(H, 1)))


function prepare_sample_target(θ₀, ℓπ; hessian_approx=:exact)
    Vfunc = x -> -ℓπ(x) # potential energy is the negative log-probability
    _Hfunc = @match hessian_approx begin # x -> (value, gradient, hessian)
        :exact      => gen_hess_rev(Vfunc, θ₀)
        :diag       => gen_hess_hybrid_diag(Vfunc, θ₀)
    end
    Hfunc = x -> copy.(_Hfunc(x)) # _Hfunc do in-place computation, copy to avoid bug

    fstabilize = H -> H #+ hps.λ * I
    Gfunc = x -> begin
        H = fstabilize(Hfunc(x)[3])
        any(.!(isfinite.(H))) ? _unit_matrix(H) : H
    end

    gen_hess = @match hessian_approx begin
        :exact      => gen_hess_fwd
        :diag       => gen_hess_fwd_diag
    end
    
    _∂G∂θfunc = @match hessian_approx begin
        :exact      => gen_∂G∂θ_fwd(Vfunc, θ₀; f = fstabilize, gen_hess = gen_hess) # size==(4, 2)
        :diag       => gen_∂G∂θ_fwd_diag(Vfunc, θ₀; f = fstabilize, gen_hess = gen_hess) # size==(2, 2)
    end
    ∂G∂θfunc = @match hessian_approx begin
        :exact      => x -> reshape_∂G∂θ(_∂G∂θfunc(x)) # size==(2, 2, 2)
        :diag       => _∂G∂θfunc # size==(2, 2)
    end

    return Vfunc, Hfunc, Gfunc, ∂G∂θfunc
end
