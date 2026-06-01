#################################################################
# Function for trajectory generation
#################################################################

function generate_full_trajectory(
        ℓπ, 
        metric_name::Symbol, 
        seed::Int,
        n_steps::Int = 5,
        step_size::Float64 = 0.1;
        init_θ = nothing,
        init_r = nothing,
        cache::Bool = true,
        a = 5.0,
        inv_mass_mat = nothing,
        relativistic_params = nothing,
        max_fp = 50,
        fp_tol = 0.01
    )
    GC.gc()

    D = ℓπ.dim
    rng = SplittableRandom(seed)

    # set initial state
    if isnothing(init_θ)
        if isa(ℓπ, LogisticRegression) 
            if isa(ℓπ.prior, HorseshoePrior)
                init_θ = vcat(randn(rng,Int(floor(ℓπ.dim / 2))),
                    exp.(randn(rng,Int(ℓπ.dim / 2))))
            else
                init_θ = randn(rng,ℓπ.dim)
            end
        else
            init_θ = randn(rng,ℓπ.dim)
        end
    end

    # create hamiltonian
    if a == Inf
        map = IdentityMap()
    else
        map = SoftAbsMap(a)
    end
    metric = create_metric(metric_name, ℓπ, D; cache, map, a, inv_mass_mat, initstate = init_θ)
    if isnothing(relativistic_params)
        h = Hamiltonian(metric, ℓπ, AutoEnzyme())
    else
        kine = RelativisticKinetic(relativistic_params...)
        h = relHamiltonian(metric, kine, ℓπ, AutoEnzyme())
    end

    if isnothing(init_r) # sample from kinetic dist if not specified
        init_r = AdvancedHMC.rand_momentum(rng,h.metric,h.kinetic,init_θ)
    end


    # set integrator
    if metric_name == :diag_rie
        lf = ImplicitMidpoint(step_size,max_fp,fp_tol)
    else
        lf = Leapfrog(step_size)
    end

    # generate full trajectory
    rng = SplittableRandom(seed)
    z_init = AdvancedHMC.phasepoint(h,init_θ,init_r)
    z_temp = copy(z_init)

    z_new_vec = AdvancedHMC.step(lf, h, z_temp, n_steps; full_trajectory = Val(true))
    z_new_vec = vcat(z_init,z_new_vec)        

    # store results
    θ_mat = Matrix{Float64}(undef, n_steps + 1, D)
    θ_mat[1,:] = init_θ
    r_mat = Matrix{Float64}(undef, n_steps + 1, D)
    r_mat[1,:] = init_r
    for t in 1:n_steps
        θ_mat[t+1,:] = z_new_vec[t+1].θ
        r_mat[t+1,:] = z_new_vec[t+1].r
    end

    energy_vec = [-AdvancedHMC.neg_energy(z_new_vec[i]) for i in eachindex(z_new_vec)]

    if ((metric_name == :diag_rie) | (metric_name == :coord_rie)) 
        n_steps = h.metric.fp_count[1]
    end

    # time trajectory run
    z_temp = copy(z_init)
    res = @timed AdvancedHMC.step(lf, h, z_temp, n_steps)

    return z_new_vec, θ_mat, r_mat, energy_vec, res.time, n_steps, h
end
