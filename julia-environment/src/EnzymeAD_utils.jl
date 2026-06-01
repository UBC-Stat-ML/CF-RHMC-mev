#################################################################
# Main functions
#################################################################

# dU_i/d_xi
function first_derivative(t, params, temp, i, x)
   return autodiff(Enzyme.Forward, local_logd, Const(t), Const(params), Const(temp), Const(i), Duplicated(x, 1.))[1]
end

# d^2 U_i / dx_i^2
function second_derivative(t, params, temp, i, x)
   deriv(x) = autodiff(Enzyme.Forward, local_logd, Const(t), Const(params), Const(temp), Const(i), Duplicated(x, 1.))[1]
   return autodiff(Enzyme.Forward, deriv, Duplicated(x, 1.))[1]
end

# metric
function G_mat(t, q::AbstractArray{T}) where {T}
   temp = get_temp(t,q)
   G_mat = ones(length(q))
   for i in eachindex(q) 
      G_mat[i] = - second_derivative(t, q, temp, i, q[i])
   end
   return G_mat
end

# log det metric
function det_G(t, q::AbstractArray{T}, p::AbstractArray{T}; dual_val = false) where {T} 
   temp = get_temp(t,q)
   result = zero(T)
   for i in eachindex(q) 
      G = (abs(- second_derivative(t, q, temp, i, q[i])))
      G_inv = 1/(coth(5*G)*G)
      result += (- log(G_inv)) / 2 
   end
   return result
end

# gaussian kinetic w/o log det
function p_invG_p(t, q::AbstractArray{T}, p::AbstractArray{T}; dual_val = false) where {T} 
   temp = get_temp(t,q)
   result = zero(T)
   for i in eachindex(q) 
      G = (abs(- second_derivative(t, q, temp, i, q[i])))
      G_inv = 1/(coth(5*G)*G)
      result += (p[i]^2 * G_inv) / 2 
   end
   return result
end


# gaussian kinetic
function kinetic(t, q::AbstractArray{T}, p::AbstractArray{T}, a = 5.0; dual_val = false) where {T} 
   temp = get_temp(t,q)
   if dual_val
      result = zero(T)
      G_inv = ones(length(q),1)
      G_mat = ones(length(q),1)
      for i in eachindex(q) 
         G = (abs(- second_derivative(t, q, temp, i, q[i])))
         if a == Inf
            G_inv[i,1] = 1/G
         else
            G_inv[i,1] = 1/(coth(a*G)*G)            
         end
         result += (p[i]^2 * G_inv[i,1] - log(G_inv[i,1])) / 2 
         G_mat[i,1] = abs(- second_derivative(t, q, temp, i, q[i]))
      end
      # return kinetic and mass matrix diagonal
      return AdvancedHMC.DualValue([result], G_mat) 
   else
      result = zero(T)
      for i in eachindex(q) 
         G = (abs(- second_derivative(t, q, temp, i, q[i])))
         if a == Inf
            G_inv = 1/G
         else
            G_inv = 1/(coth(a*G)*G)            
         end
         result += (p[i]^2 * G_inv - log(G_inv)) / 2 
      end
      return result
   end
end

# relativistic kinetic
function rel_kinetic(t, q::AbstractArray{T}, p::AbstractArray{T}, m, c, a = 5.0; dual_val = false) where {T} 
   temp = get_temp(t,q)
   if dual_val
      result1 = zero(T)
      result2 = zero(T)
      G_inv = ones(length(q),1)
      G_mat = ones(length(q),1)
      for i in eachindex(q) 
         G = (abs(- second_derivative(t, q, temp, i, q[i])))
         if a == Inf
            G_inv[i,1] = 1/G
         else
            G_inv[i,1] = 1/(coth(a*G)*G)            
         end
         result1 += sqrt(p[i]^2 * G_inv[i,1] / (m^2 * c^2) + 1) * m * c^2 
         result2 -= log(G_inv[i,1]) / 2

         G_mat[i,1] = abs(- second_derivative(t, q, temp, i, q[i]))
      end
      # return kinetic and mass matrix diagonal
      return AdvancedHMC.DualValue([result1 + result2], G_mat) 
   else
      result1 = zero(T)
      result2 = zero(T)
      for i in eachindex(q) 
         G = (abs(- second_derivative(t, q, temp, i, q[i])))
         if a == Inf
            G_inv = 1/G
         else
            G_inv = 1/(coth(a*G)*G)            
         end
         result1 += sqrt(p[i]^2 * G_inv / (m^2 * c^2) + 1) * m * c^2 
         result2 -= log(G_inv) / 2
      end
      return result1 + result2
   end
end


# hamitonial
hamiltonian(target, metric, q, p, a = 5.0) = -logdensity(target, q) + kinetic(metric, q, p, a)
rel_hamiltonian(target, metric, q, p, m, c, a = 5.0) = -logdensity(target, q) + rel_kinetic(metric, q, p, m, c, a)

#################################################################
# Enzyme AD utilities for glm
#################################################################

function local_logprior(t::LogisticRegression{<:NormalPrior}, params, temp, i, value)
   return temp[end] - (value^2 - params[i]^2) / (2 * 100)#prior_temp.prior_scales[i])
   # default normal prior scales to 100 for now since calling prior_temp.prior_scales[i]
   # causes problems for Enzyme
end

function local_logprior(t::LogisticRegression{<:HorseshoePrior}, params, temp, i, value)
   res = 0.0
   d = Int(floor(t.dim / 2))
   if i == 1 # first coefficient
      # 2log(1 + β1^2/3)
      res += temp[end] - 2*log(1+value^2/3) + 2*log(1+params[i]^2/3)
   elseif i <= d # other coefficients
      # βi^2/(2*(λi * τ)^2))
      res += temp[end] - value^2/(2*(params[i+d-1]*params[end])^2) + params[i]^2/(2*(params[i+d-1]*params[end])^2)
   elseif i < 2*d # local scales
      # log(λi^2)/2 + log(λi^2+1) + βi^2/(2*(λi * τ)^2)
      res += temp[end] - (params[i-d+1]^2/(2*(value*params[end])^2) + log((value)^2)/2 + log(1+value^2)) +
      (params[i-d+1]^2/(2*(params[i]*params[end])^2) + log((params[i])^2)/2 + log(1+params[i]^2))
   else # global scale
      # (∑ βi^2/(2 λi^2) / τ^2 + (d-1)/2 * log(τ^2) + log(τ^2+1)
      sum_βλ = sum_βλ_ratio_square(params,d)
      res += temp[end] - sum_βλ/value^2 - (d-1)*log(value^2)/2 - log(value^2+1) +
            (sum_βλ/params[i]^2 + (d-1)*log(params[i]^2)/2 + log(1+params[i]^2))
   end

   return res
end

function local_logprior(t::LogisticRegression{<:ModifiedHorseshoePrior}, params, temp, i, value)
   res = 0.0
   d = Int(floor(t.dim / 2))
   if i == 1 # first coefficient
      # 2log(1 + β1^2/3)
      res += temp[end] - 2*log(1+value^2/3) + 2*log(1+params[i]^2/3)
   elseif i <= d # other coefficients
      # βi^2/(2*e^(λi + τ))
      res += temp[end] - value^2/(2*exp(params[i+d-1] + params[end])) + params[i]^2/(2*exp(params[i+d-1] + params[end]))
   elseif i < 2*d # local scales
      # log(e^λi+1) + βi^2/(2*e^(λi + τ))
      res += temp[end] - (params[i-d+1]^2/(2*exp(value + params[end])) + log(1+exp(value))) +
      (params[i-d+1]^2/(2*exp(params[i] + params[end])) + log(1+exp(params[i])))
   else # global scale
      # (∑ βi^2/(2 e^λi) / e^τ + (d-2)/2 * τ + log(e^τ+1)
      sum_βλ = sum_βλ_ratio_exp(params,d)
      res += temp[end] - sum_βλ/exp(value) - (d-2)*value/2 - log(exp(value)+1) +
            (sum_βλ/exp(params[i]) + (d-2)*params[i]/2 + log(exp(params[i])+1))
   end

   return res
end


function local_loglikelihood(t::LogisticRegression, params, temp, i, value)
   res = 0.0
   for j in eachindex(t.vec_y)
      xβ = temp[j] + (value - params[i]) * t.rows_x[j][i]
      res -= log1pexp(- (2*t.vec_y[j]-1) * xβ)
   end      
   return res
end

function local_loglikelihood(t::LogisticRegression{<:HierarchicalPrior}, params, temp, i, value)
   res = 0.0
   d = Int(floor(t.dim / 2))
   if i <= d
      for j in eachindex(t.vec_y)
         xβ = temp[j] + (value - params[i]) * t.rows_x[j][i]
         res -= log1pexp(- (2*t.vec_y[j]-1) * xβ)
      end
   else
      res = temp[end-1]
   end
   return res
end

function local_logd(t::LogisticRegression, params, temp, i, value)
   # log prior
   # Note: last element of temp stores full log prior
   res = local_logprior(t, params, temp, i, value)

   # log likelihood
   res += local_loglikelihood(t, params, temp, i, value)

   return res
end

function logdensity(t::LogisticRegression, params::AbstractArray{T}) where {T} 
   res = 0.0

   # prior
   res += log_prior(t,params)

   # log likelihood, vec_y has values 0,1
   res += log_likelihood(t,params)

   return res
end

function get_temp(t::LogisticRegression, params::AbstractArray{T}) where {T}
   temp = Vector{T}(undef, length(t.vec_y)+1)
   for j in eachindex(t.vec_y)
      temp[j] = dot(t.rows_x[j],params)
   end

   temp[end] = log_prior(t,params)

   return temp
end

function get_temp(t::LogisticRegression{<:HierarchicalPrior}, params::AbstractArray{T}) where {T}
   # store full log likelihood at end-1 index
   temp = Vector{T}(undef, length(t.vec_y)+2)
   log_lik = 0.0
   d = Int(floor(t.dim / 2))
   for j in eachindex(t.vec_y)
      temp[j] = dot(t.rows_x[j],params[1:d])
      log_lik -= log1pexp(- (2*t.vec_y[j]-1) * temp[j])
   end

   temp[end-1] = log_lik 
   temp[end] = log_prior(t,params)
   
   return temp
end



#################################################################
# Enzyme AD utilities for Funnel target
#################################################################

function local_logd(t::FunnelTarget, params, temp, i, value)
   # temp store x[2:end]^T x[2:end]

   if i == 1
      res = - (t.dim - 1) * value / (2 * t.b) - temp / (2 * exp(value / t.b)) - value^2/18
   else
      res = - (t.dim - 1) * params[1] / (2 * t.b) - (temp + value^2 - params[i]^2) / (2 * exp(params[1] / t.b)) - params[1]^2/18
   end

   return res
end

function logdensity(t::FunnelTarget, x::AbstractArray{T}) where {T} 
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

function get_temp(t::FunnelTarget, params::AbstractArray{T}) where {T}
   return dot(params ,params) - params[1]^2
end


#################################################################
# Enzyme AD utilities for independent targets
#################################################################

function local_logd(t, params, temp, i, value)
    term_new = @inline component_logd(t,i,value)
    term_old = @inline component_logd(t,i,params[i])
    return term_new - term_old + temp[end]
end

function logdensity(t, params::AbstractArray{T}) where {T} 
    res = zero(T)
    for i in eachindex(params)
        res += @inline component_logd(t, i, params[i])
    end
    return res
end

get_temp(t, params::AbstractArray{T}) where {T} = [logdensity(t,params)]

#################################################################
# correctness check for local_logd and its second derivative
#################################################################
function check_local_logd(t)
   params = randn(t.dim)
   for i in eachindex(params)
      new_value = randn()
      new_params = copy(params)
      new_params[i] = copy(new_value)

      full_logd = logdensity(t,new_params)
      temp = get_temp(t,params)
      coord_logd = local_logd(t,params,temp,i,new_value)

      fi = function(xi)
         new_x = vcat(params[1:(i-1)],xi,params[(i+1):end])
         return logdensity(t,new_x)
      end

      hess_i = FiniteDiff.finite_difference_hessian(fi,[new_value])[1,1]
      coord_hess_i = second_derivative(t,params,temp,i,new_value)

      @assert full_logd ≈ coord_logd
      @assert isapprox(hess_i, coord_hess_i; rtol = 0.01)
   end
end


#################################################################
# timing ADs for Enzyme
#################################################################

function timings(t)
   params = randn(t.dim)

   println("density evaluation")
   @time logdensity(t, params);

   # just reverse mode autodiff of the target alone
   grad_q, grad_p = zeros(t.dim), zeros(t.dim)
   println("AD of logdensity (Enzyme Reverse Diff)")
   @time autodiff(Enzyme.Reverse, logdensity, Active, Const(t), Duplicated(params, grad_q))

   # now AD on the hamiltonian 
   println("AD of Hamiltonian with diagonal Hessian based metric (Enzyme Reverse over Forward Diff)")
   @time autodiff(Enzyme.Reverse, hamiltonian, Active, Const(t), Const(t), Duplicated(params, grad_q), Duplicated(params, grad_p))

end
