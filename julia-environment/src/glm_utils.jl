#################################################################
# GLM utility functions
#################################################################

is_one(x) = Int(x == 1 ? 1 : 0)

# function to convert csv to mat file (so that load_and_preprocess_data goes smoothly)
function csv_to_mat_logistic(data_name, y_col)
   csv_path = joinpath(dirname(base_dir()), "data", data_name * ".csv")
   data_csv = DataFrame(CSV.File(csv_path))
   if y_col == :end
      y_col = size(data_csv,2)
   end
   idx = (1:size(data_csv,2)) .!== y_col
   y_vec = Float64.(Array(data_csv[:,y_col:y_col] .== data_csv[1,y_col]))
   data_dict = Dict("X" => Array(data_csv[:,idx]), "Y" => y_vec)
   mat_path = joinpath(dirname(base_dir()), "data", data_name * ".mat")

   matwrite(mat_path,data_dict)
end

# function for loading (or creating) and preprocessing glm data
function load_and_preprocess_data(
   data_name;
   distribution = false, # true when sampling from a distribution
   synthetic = true,
   model = "logistic",
   generated_dim = 2^15,
   generated_size = 2^8,
   requested_dim = nothing, 
   requested_size = nothing, 
   rng = SplittableRandom(1), 
   add_intercept = true,
   binarize = is_one,
   standardize = true,
   true_params = nothing,
   design_mat = nothing,
   random_col = true,
   random_subset = false,
   sparse_standardize = true
   )

   if distribution
      return [nothing], [nothing], 0, requested_dim
   else
      if synthetic
         data_sim_rng = SplittableRandoms.split(rng)
         if !isnothing(true_params)
            generated_dim = size(true_params)[1]-1 # set gen dim to true_params dim
            if !isnothing(design_mat)
               @assert generated_dim == size(design_mat)[2] # check of true_params and design_mat have conforming sizes
               generated_size = size(design_mat)[1] # set gen size to design_mat size
               data, true_params = create_data(data_sim_rng,generated_size,generated_dim,true_params,design_mat; model = model)
            else
               data, true_params = create_data(data_sim_rng,generated_size,generated_dim,true_params; model = model)
            end
         else # no need to control design_mat when param is not controlled
            data, true_params = create_data(data_sim_rng,generated_size,generated_dim; model = model)
         end
      else
         data = matread(joinpath(dirname(base_dir()), "data", data_name * ".mat"))
      end

      mat_x = data["X"]
      vec_y = data["Y"][:, begin]
      original_data_size, original_data_dim = size(mat_x)

      dim_rng = SplittableRandoms.split(rng)
      if !isnothing(requested_dim)
         @assert requested_dim ≤ original_data_dim
         
         if random_col
            d_idx = randperm(dim_rng, original_data_dim)[1:requested_dim]
         else
            if random_subset
               d_idx = randperm(dim_rng, requested_dim)
            else
               d_idx = 1:requested_dim
            end
         end
         mat_x = mat_x[:, d_idx]
      end

      instances_rng = SplittableRandoms.split(rng)
      if !isnothing(requested_size)
         @assert requested_size ≤ original_data_size
         n_idx = randperm(instances_rng, original_data_size)[1:requested_size]
         mat_x = mat_x[n_idx, :]
         vec_y = vec_y[n_idx]
      end
      
      data_size, data_dim = size(mat_x)
      if standardize
         n_zeros = sum(x -> x == 0 ? 1 : 0, mat_x) 
         sparsity = n_zeros/data_size/data_dim
         for c in eachcol(mat_x)
            s = std(c)
            if s > 0
               if (sparsity > 0.85) & sparse_standardize # when sparsity is high, only scale down the none 0 entries
                  idx = (c .!= 0)
                  c[idx] .= c[idx] ./ maximum(abs.(c[idx])) # use max abs scaling so features lie in [-1,1]
               else
                  c .= (c .- mean(c)) ./ s
               end
            else
               @warn "Trying to standardize data with zero standard deviation" maxlog=1
            end
         end
      end
      if add_intercept # after standardization!
         mat_x = hcat(ones(data_size), mat_x) 
      end

      if model == "linear"
         converted_y = vec_y
      elseif model == "logistic"
         # all data with 3 classes or more default to the same format -> no need to change
         # data with 2 classes have different formats (-1,1 or 1,2, or 0,1) -> change to 0,1
         converted_y = isnothing(binarize) ? Int.(vec_y) : binarize.(vec_y)
      elseif model == "categorical" || model == "poisson"
         converted_y = Int.(vec_y)
      else
         error()
      end
      rows_x   = collect(eachrow(mat_x)) # loglik only looks at rows, so this improves efficiency
      data_size, data_dim = size(mat_x)
      @assert length(rows_x) == data_size
      @assert length(first(rows_x)) == data_dim

      return rows_x, converted_y, data_size, data_dim
   end
end


# function for creating glm data sets
function create_data(rng, n, d, β = randn(rng, d+1), x_mat = randn(rng, n, d); model = "logistic", noise = 0.1, num_class = 3)
   if model == "logistic"
      intercept = fill(β[1],n)
      p_vec = 1 ./ (1 .+ exp.( .- intercept .- x_mat*β[2:end]))
      y_vec = rand(rng, n) .< p_vec
   elseif model == "linear"
      intercept = fill(β[1],n)
      y_vec = intercept .+ x_mat*β[2:end] .+ noise .* randn(rng, n) # noise = sd of obs noise
   elseif model == "categorical"
      if length(β) != (d+1)*num_class
         β = randn(rng, d+1, num_class)         
      end
      xβ = hcat(ones(n), x_mat)*β
      p_vec = LogExpFunctions.softmax(xβ)
      y_vec = [sample(rng,1:num_class,Weights(xβ[i,:])) for i in 1:n]
   end

   # conform with real dataset format:
   return Dict("X" => x_mat, "Y" => y_vec), β
end

#################################################################
# utility calculation for horseshoe prior
#################################################################

# function to return ∑ βi^2/(2 * λi^2)
function sum_βλ_ratio_square(params,d)
   res = 0.0
   for i in 2:d
      res += params[i]^2/(2*(params[i+d-1])^2)
   end
   return res
end

# function to return ∑ βi^2/(2 * e^λi)
function sum_βλ_ratio_exp(params,d)
   res = 0.0
   for i in 2:d
      res += params[i]^2/(2*exp(params[i+d-1]))
   end
   return res
end


#################################################################
# naive conditional density and its hessian
#################################################################

function naive_ℓπi_hessian(xi,θ,i,rows_x,vec_y,prior_σ²)
   θ_new = vcat(θ[1:(i-1)],xi,θ[(i+1):end])
   n = length(vec_y)
   xy_vec = [((2*vec_y[j]-1)*rows_x[j][i])^2 for j in 1:n]
   exy = Vector{Float64}(undef, n)
   for j in 1:n
      exy[j] = exp((2*vec_y[j]-1)*dot(θ_new,rows_x[j]))
   end

   return sum((xy_vec .* exy) ./ (exy .+ 1).^2) + 1/prior_σ²
end

function naive_neg_energy(h,r,θ,rows_x,vec_y,prior_σ²)
   D = size(h.metric)[1]
   v = [naive_ℓπi_hessian(θ[i],θ,i,rows_x,vec_y,prior_σ²) for i in 1:D]
   # @show r
   # @show θ
   # @show v
   return - 1 / 2 * (D * log(2π) + sum(log.(v))) - sum( r .^ 2 ./ v) / 2
end

function cg_ℓπi_hessian(xi,θ,i,rows_x,vec_y,prior_σ²,metric)
   n = length(vec_y)
   res = 0.0
   xy_vec = [((2*vec_y[j]-1)*rows_x[j][i])^2 for j in 1:n]
   exy = Vector{Float64}(undef, n)
   for j in 1:n
      xy² = ((2*vec_y[j]-1)*rows_x[j][i])^2
      βxy = (2*vec_y[j]-1)*((xi[1]-θ[i])*rows_x[j][i] + metric.temp[j])
      res += logistic(βxy)*logistic(-βxy)*xy²
   end

   res += 1/prior_σ²

   return res
end
