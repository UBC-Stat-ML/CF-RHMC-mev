#################################################################
# Metric generation function
#################################################################

function create_metric(
   metric_name::Symbol,
   ℓπ,
   dimension::Int;
   cache::Bool = false,
   map = IdentityMap(),
   a = 5.0,
   inv_mass_mat = nothing,
   initstate = nothing
   )

   if metric_name == :diag_rie
      if cache & isa(ℓπ,LogisticRegression)
         return DiagRiemannianMetric((dimension,),a,ℓπ,[0],[0])
      elseif cache # for coord-wise indep targets
         return DiagRiemannianMetric((dimension,),a,ℓπ,[0],[0])
      else # use dense riemannian instead
         f = function (θ)
            LogDensityProblems.logdensity(ℓπ,θ)
         end

         _, _, G, ∂G∂θ = prepare_sample_target(initstate, f; hessian_approx=:diag)

         return DenseRiemannianMetric((dimension,),G,∂G∂θ,map)
      end
   elseif metric_name == :dense
      if isnothing(inv_mass_mat)
         return DenseEuclideanMetric(dimension)
      else
         return DenseEuclideanMetric(inv_mass_mat)
      end
   elseif metric_name == :diagonal
      if isnothing(inv_mass_mat)
         return DiagEuclideanMetric(dimension)
      else
         if !isa(inv_mass_mat,Vector) # convert to a vector if is a matrix
            inv_mass_mat = diag(inv_mass_mat)
         end
         return DiagEuclideanMetric(inv_mass_mat)
      end
   elseif metric_name == :unit
      return UnitEuclideanMetric(dimension)
   end
end
