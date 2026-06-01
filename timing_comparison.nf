
include { crossProduct; filed; collectCSVs; combine_process; instantiate; head; pow; deliverables; findDataDim; estimateMemoryMinESS } from './utils.nf'

def variables = [
    seed: 1..20,
]


// global variables
params.dryRun = false

// local variables
def julia_env_dir = file("julia-environment")
def julia_depot_dir = file("$JULIA_DEPOT_PATH")
def deliv = deliverables(workflow)

workflow {
    args = crossProduct(variables, params.dryRun)
    julia_env = instantiate(julia_depot_dir, julia_env_dir) //| precompile
    agg_path = runSimulation(julia_depot_dir, julia_env, args) 
    combine_process(julia_env, agg_path.collect())
}

process runSimulation {
    debug true
    memory { 20.GB * Math.pow(2, task.attempt-1) }
    time { 60.hour } 
    cpus 1
    errorStrategy 'ignore' //{ task.attempt < 3 ? 'retry' : 'ignore' } 
    input:
        env JULIA_DEPOT_PATH
        path julia_env
        val arg
    output:
        path "${filed(arg)}"
    """
    #!/usr/bin/env -S julia --heap-size-hint=${task.memory.toGiga()}G

    using Pkg
    Pkg.activate(joinpath("$baseDir", "$julia_env")) 
    include(joinpath("$baseDir", "$julia_env", "src", "trajectory_utils.jl")) # loads sampling functions
    include(joinpath("$baseDir", "$julia_env", "src", "plotting_functions.jl")) # loads plotting functions

    function main()
        # collect global vars 
        seed = 6435
        sampler_name = :NUTS
        # change to 6 fp iter per step to count grad calls
        sampler_specs = (; metric_name = :diag_rie, max_fp = 6)
        target_name = :glm

        max_dim = 11

        # test run to get rid of compile time
        @info "Starting prelim run for code compilation"
        for i in 1:3
            GC.gc()

            target_specs = (; dim = Int(2^i), n = 3, synthetic = true)
            ℓπ = build_target(target_name, seed; target_specs...)

            n_steps = 10
            z_vec_Diag, θ_mat_Diag, r_mat_Diag, energy_Diag, time_Diag, ns_diag, h_diag = generate_full_trajectory(ℓπ, :diag_rie, seed, n_steps, 0.1; cache = false);
            z_vec_Diag2, θ_mat_Diag2, r_mat_Diag2, energy_Diag2, time_Diag2, ns_diag2, h_diag2 = generate_full_trajectory(ℓπ, :diag_rie, seed, n_steps, 0.1; cache = true);
        end

        @info "Starting timed run"
        time_no_cache = []
        n_steps_no_cache = []
        time_cache = []
        n_steps_cache = []
        for i in 1:max_dim
            GC.gc()
            @show i

            target_specs = (; dim = Int(2^i), n = 3, synthetic = true)
            ℓπ = build_target(target_name, seed; target_specs...)

            n_steps = 2
            z_vec_Diag, θ_mat_Diag, r_mat_Diag, energy_Diag, time_Diag, ns_diag, h_diag = generate_full_trajectory(ℓπ, :diag_rie, seed, n_steps, 0.1; cache = false);
            z_vec_Diag2, θ_mat_Diag2, r_mat_Diag2, energy_Diag2, time_Diag2, ns_diag2, h_diag2 = generate_full_trajectory(ℓπ, :diag_rie, seed, n_steps, 0.1; cache = true);

            push!(time_no_cache,time_Diag)
            push!(n_steps_no_cache,ns_diag)
            push!(time_cache,time_Diag2)
            push!(n_steps_cache,ns_diag2)
        end

        @info "Finished"

        df = DataFrame(dimension = 2 .^ (1:max_dim), time_cache = time_cache, 
            time_no_cache = time_no_cache, n_steps_cache = n_steps_cache, n_steps_no_cache = n_steps_no_cache)
        
        isdir("${filed(arg)}") || mkdir("${filed(arg)}")
        CSV.write("${filed(arg)}/summary.csv", df)
    end

    main()
    """
}


