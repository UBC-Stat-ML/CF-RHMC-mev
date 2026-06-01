#################################################################
# Pigeons utilities
#################################################################

using Pigeons

#### glm target w/ unconstrained parametrization
# log potential
function (log_potential::LogisticRegression{<:ModifiedHorseshoePrior})(params)
    d = Int(length(params) / 2)
    res = 0.0
    # log likelihood, vec_y has values 0,1
    for j in eachindex(log_potential.vec_y)
        xβ = dot(log_potential.rows_x[j],params[1:d])
        res -= log1pexp(- (2*log_potential.vec_y[j]-1) * xβ)
    end

    # log prior
    res += - 2*log(1+params[1]^2/3) - log(1+exp(params[end])) - (d-2)/2*params[end]
    for i in 2:d
        res += - params[i]^2 / (2*exp(params[i+d-1] + params[end])) - log(1 + exp(params[i+d-1]))
    end

    return res
end

Pigeons.initialization(t::LogisticRegression{<:ModifiedHorseshoePrior}, rng::AbstractRNG, ::Int) = randn(rng,t.dim)

function Pigeons.sample_iid!(::LogisticRegression{<:ModifiedHorseshoePrior}, replica, shared)
    state = replica.state
    d = Int(length(state) / 2)
    rng = replica.rng
    β0 = rand(rng,TDist(3))
    τ = log(rand(rng,Cauchy())^2)
    λ = log.(rand(rng,Cauchy(),d) .^ 2)
    β = [rand(rng,Normal(0,exp(λ[i]+τ))) for i in 1:(d-1)]
    return vcat(β0,β,λ,τ)
end
