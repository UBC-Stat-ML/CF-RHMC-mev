
include { crossProduct; filed; collectCSVs; combine_process; instantiate; head; pow; deliverables; findDataDim; estimateMemoryMinESS } from './utils.nf'

def variables = [
    seed: 1..20,
    sampler: ["NUTS"],
    kinetic: ["gaussian"],
    target: ["glm","funnel"],
    dimension: (1..7).collect{pow(2,it)},
]

regularization = [ 
    glm: "5",
    funnel: "Inf",
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
    memory { 20.GB }
    time { 20.hour } 
    cpus 1
    errorStrategy 'ignore'
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
    include(joinpath("$baseDir", "$julia_env", "src", "sampling_functions.jl")) # loads sampling functions
    include(joinpath("$baseDir", "$julia_env", "src", "plotting_functions.jl")) # loads plotting functions

    function main()
        rnuts_specs = (; metric_name = :diag_rie, a = ${regularization[arg.target]})

        # target specs
        target_seed = 675
        target_name = :${arg.target}
        if target_name == :glm
            target_specs = (;dim = ${arg.dimension}, n = 20, prior_type = :modifiedhorseshoe, synthetic = true)
        elseif target_name == :funnel
            target_specs = (;dim = ${arg.dimension}, b = 0.5)
        end
        
        # RNUTS chains
        sampler_seed = ${arg.seed}
        n = 10000

        chain_RNUTS, runtime_RNUTS, adapttime_RNUTS, n_steps_RNUTS, h_RNUTS = sample_chain(sampler_seed, target_seed, n;
            sampler_specs = rnuts_specs, target_name, target_specs, return_chain = true)

        df_RNUTS_stats = DataFrame(metric = :diag_rie, 
            runtime = runtime_RNUTS, adapttime = adapttime_RNUTS, 
            integration_steps = sum(chain_RNUTS[:n_steps][(n+1):end]), 
            gradient_calls = chain_RNUTS[:gradient_calls][end] - chain_RNUTS[:gradient_calls][n], 
            integrator_fails = chain_RNUTS[:integrator_fails][end] - chain_RNUTS[:integrator_fails][n],
            numerical_error = sum(chain_RNUTS[:numerical_error][(n+1):end]),
            acc_rate = mean(chain_RNUTS[:acceptance_rate][(n+1):end]),
            step_size = chain_RNUTS[:step_size][end]
        )
        
        isdir("${filed(arg)}") || mkdir("${filed(arg)}")
        CSV.write("${filed(arg)}/summary_RNUTS.csv", df_RNUTS_stats)
    end

    main()
    """
}
