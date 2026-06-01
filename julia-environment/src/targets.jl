using LogDensityProblems

#################################################################
# logistic regression target
#################################################################
abstract type PriorDist end
abstract type HierarchicalPrior <: PriorDist end

@auto struct NormalPrior <: PriorDist
    prior_scales
end

# horseshoe prior: 2d params
# y ~ logistic(βx)
# β1 ~ t(3,0,1)
# βi ~ N(0, (λi * τ)^2)
# λi, τ ~ C(0,1)
struct HorseshoePrior <: HierarchicalPrior end

# modified horseshoe prior: 2d params
# y ~ logistic(βx)
# β1 ~ t(3,0,1)
# βi ~ N(0, e^(λi + τ))
# λi, τ ~ log(C(0,1)^2)
struct ModifiedHorseshoePrior <: HierarchicalPrior end


struct LogisticRegression{P<:PriorDist,X,Y}
    prior::P
    dim::Int
    rows_x::X
    vec_y::Y
end

### log prior function for each kind of prior
function log_prior(t::LogisticRegression{<:NormalPrior}, params::AbstractArray{T}) where {T} 
   return - dot( params ./ (2*t.prior.prior_scales), params)
end

function log_prior(t::LogisticRegression{<:HorseshoePrior}, params::AbstractArray{T}) where {T}
    d = Int(t.dim / 2)
    res = - 2*log(1+params[1]^2/3) - log(1+params[end]^2) - (d-1)/2*log(params[end]^2)
    for i in 2:d
        res += - params[i]^2/(2*(params[i+d-1]*params[end])^2) - log(1 + params[i+d-1]^2) - log(params[i+d-1]^2)/2
    end
    return res
end

function log_prior(t::LogisticRegression{<:ModifiedHorseshoePrior}, params::AbstractArray{T}) where {T}
    d = Int(t.dim / 2)
    res = - 2*log(1+params[1]^2/3) - log(1+exp(params[end])) - (d-2)/2*params[end]
    for i in 2:d
        res += - params[i]^2 / (2*exp(params[i+d-1] + params[end])) - log(1 + exp(params[i+d-1]))
    end
    return res
end

### log likelihood
function log_likelihood(t::LogisticRegression, params::AbstractArray{T}) where {T} 
    res = 0.0
    for j in eachindex(t.vec_y)
        xβ = dot(t.rows_x[j],params)
        res -= log1pexp(- (2*t.vec_y[j]-1) * xβ)
    end
    return res
end

function log_likelihood(t::LogisticRegression{<:HierarchicalPrior}, params::AbstractArray{T}) where {T} 
    res = 0.0
    d = Int(floor(t.dim / 2))
    for j in eachindex(t.vec_y)
        xβ = dot(t.rows_x[j],params[1:d])
        res -= log1pexp(- (2*t.vec_y[j]-1) * xβ)
    end
    return res
end



# Log density
function LogDensityProblems.logdensity(t::LogisticRegression, params)
    res = 0.0
    # log likelihood, vec_y has values 0,1
    res += log_likelihood(t,params)

    # log prior
    res += log_prior(t,params)

    return res
end

LogDensityProblems.dimension(p::LogisticRegression) = p.dim
# Declare that the log density function is defined
function LogDensityProblems.capabilities(::LogisticRegression)
    return LogDensityProblems.LogDensityOrder{0}()
end

# glm target generation
function glm_target(
        seed; 
        n = nothing, 
        dim = nothing,
        prior_type = :normal,
        prior_params = nothing,
        data_name = "colon",
        synthetic = false,
        generated_dim = 2^15,
        generated_size = 2^8,
        true_params = nothing,
        design_mat = nothing,
    )
    rng = SplittableRandom(seed)
    rows_x, vec_y, data_size, D = load_and_preprocess_data(data_name; synthetic,
        requested_dim = dim, standardize = true, requested_size = n, generated_dim, generated_size,
        model = "logistic", rng, true_params, design_mat)
    
    if isnothing(prior_params)
        prior_params = repeat([100.0],D)
    end

    if prior_type == :normal
        prior_dist = NormalPrior(prior_params)
        return LogisticRegression(prior_dist, D, rows_x, vec_y)
    elseif prior_type == :horseshoe
        prior_dist = HorseshoePrior()
        return LogisticRegression(prior_dist, 2*D, rows_x, vec_y)
    elseif prior_type == :modifiedhorseshoe
        prior_dist = ModifiedHorseshoePrior()
        return LogisticRegression(prior_dist, 2*D, rows_x, vec_y)
    else
        error("prior not defined")
    end
end


#################################################################
# funnel target
#################################################################

@auto struct FunnelTarget
    dim::Int
    b # scale param, smaller is harder
end

function FunnelTarget(seed; dim = 2, b = 2.0)
    return FunnelTarget(dim,b)
end


# Log density of a multivariate logistic distribution
function LogDensityProblems.logdensity(t::FunnelTarget, x)
    res = 0.0
    for i in 1:(t.dim)
        if i == 1
            res += -x[i]^2/18 - (t.dim-1)*x[i]/(2 * t.b)
        else
            res += -x[i]^2/(2*exp(x[1] / t.b))
        end
    end
    return res
end

LogDensityProblems.dimension(p::FunnelTarget) = p.dim
# Declare that the log density function is defined
function LogDensityProblems.capabilities(::FunnelTarget)
    return LogDensityProblems.LogDensityOrder{0}()
end
