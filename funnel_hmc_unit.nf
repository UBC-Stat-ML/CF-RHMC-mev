
include { crossProduct; filed; collectCSVs; combine_process; instantiate; head; pow; deliverables; findDataDim; estimateMemoryMinESS } from './utils.nf'

def variables = [
    seed: 1..1000,
    sampler: ["hmc_unit"],
    kinetic: ["gaussian"],
    target: ["funnel"],
    dim: [100],
]

sampler_metric = [ 
    hmc_unit: "unit",
    hmc_diag: "diagonal",
    nuts_unit: "unit",
    nuts_diag: "diagonal",
    cgrhmc: "diag_rie",
    cgrnuts: "diag_rie",
    rhmc: "diag_rie",
    rnuts: "diag_rie",
]

sampler_is_nuts = [ 
    hmc_unit: "false",
    hmc_diag: "false",
    nuts_unit: "true",
    nuts_diag: "true",
    cgrhmc: "false",
    cgrnuts: "true",
    rhmc: "false",
    rnuts: "true",
]

sampler_budget = [ 
    hmc_unit: "100000",
    hmc_diag: "100000",
    nuts_unit: "15000",
    nuts_diag: "15000",
    cgrhmc: "4000",
    cgrnuts: "1000",
    rhmc: "2",
    rnuts: "2",
]

sampler_thining = [ 
    hmc_unit: "10",
    hmc_diag: "10",
    nuts_unit: "3",
    nuts_diag: "3",
    cgrhmc: "1",
    cgrnuts: "1",
    rhmc: "1",
    rnuts: "1",
]

sampler_cache = [ 
    hmc_unit: "true",
    hmc_diag: "true",
    nuts_unit: "true",
    nuts_diag: "true",
    cgrhmc: "true",
    cgrnuts: "true",
    rhmc: "false",
    rnuts: "false",
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
    time { 8.hour } 
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
    include(joinpath("$baseDir", "$julia_env", "src", "sampling_functions.jl")) # loads sampling functions
    include(joinpath("$baseDir", "$julia_env", "src", "plotting_functions.jl")) # loads plotting functions

    sampler_type = :${arg.sampler}


    function main()
        # collect global vars 
        sampler_seed = ${arg.seed}
        sampler_name = :NUTS

        metric_name = :${sampler_metric[arg.sampler]}
        nuts_id = ${sampler_is_nuts[arg.sampler]}

        target_name = :${arg.target}
        # b = 2.0 # easy
        b = 0.5 # hard
        target_specs = (; b, dim = ${arg.dim})
        target_seed = 1
        
        pilot_sampler_specs = (; metric_name, nuts_id, adapt_mode = :both, a = Inf,
                        cache = true,
                        kinetic_name = :${arg.kinetic}, L = 20)
        
        res_pilot, runtime, adapttime, evals, h_pilot = sample_chain(3522, target_seed, 10000; sampler_name, sampler_specs = pilot_sampler_specs, target_name, target_specs, return_chain = true)

        if metric_name == :diagonal
            invM = h_pilot.metric.M⁻¹
            step_size = res_pilot[:step_size][end]
            sampler_specs = (; metric_name, step_size, nuts_id, adapt_mode = :none, a = Inf,
                            cache = ${sampler_cache[arg.sampler]},
                            kinetic_name = :${arg.kinetic}, L = 20, inv_mass_mat = invM)
        else
            step_size = res_pilot[:step_size][end]
            sampler_specs = (; metric_name, step_size, nuts_id, adapt_mode = :none, a = Inf, 
                            cache = ${sampler_cache[arg.sampler]},
                            kinetic_name = :${arg.kinetic}, L = 20)
        end
        
        res, _, _, _, _ = sample_chain(sampler_seed, target_seed, ${sampler_budget[arg.sampler]}; sampler_name, sampler_specs, target_name, target_specs, return_chain = true)
        thining_gap = ${sampler_thining[arg.sampler]}
        df1 = DataFrame(Array(res[1:thining_gap:end,1,:])',:auto)
        df_time = DataFrame(Array(res[:current_time][1:thining_gap:end])' .- res[:current_time][1],:auto)
        df_steps = DataFrame(Array(res[:n_steps][1:thining_gap:end])',:auto)
        
        isdir("${filed(arg)}") || mkdir("${filed(arg)}")
        CSV.write("${filed(arg)}/first_coord.csv", df1)
        CSV.write("${filed(arg)}/timing.csv", df_time)
        CSV.write("${filed(arg)}/n_steps.csv", df_steps)
    end

    main()
    """
}
