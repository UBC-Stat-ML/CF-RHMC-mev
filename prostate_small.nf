
include { crossProduct; filed; collectCSVs; combine_process; instantiate; head; pow; deliverables; findDataDim; estimateMemoryMinESS } from './utils.nf'

def variables = [
    target: ["glm"],
    data: ["prostate_small"],
    prior: ["modifiedhorseshoe"]
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
    memory { 60.GB }
    time { 100.hour } 
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

    function main()
        # collect global vars 
        nuts_specs = (;metric_name = :diagonal)
        rnuts_specs = (; metric_name = :diag_rie)

        # target specs
        target_seed = 675
        target_name = :glm
        target_specs = (;prior_type = :modifiedhorseshoe, data_name = "${arg.data}")

        target_test = build_target(target_name, target_seed; target_specs...)
        reference_test = LogisticRegression(ModifiedHorseshoePrior(),target_test.dim,[],[])

        # NRPT chain
        pt = pigeons(target = target_test,
                    reference = reference_test,
                    n_rounds = 15,
                    record = [traces, round_trip])

        chain_NRPT = Chains(pt)

        # NUTS & RNUTS chains
        sampler_seed = 234
        n = 1000

        GC.gc()
        chain_RNUTS, runtime_RNUTS, adapttime_RNUTS, n_steps_RNUTS, h_RNUTS = sample_chain(sampler_seed, target_seed, n;
            n_rounds = 5, sampler_specs = rnuts_specs, target_name, target_specs, return_chain = true)

        GC.gc()
        chain_NUTS, runtime_NUTS, adapttime_NUTS, n_steps_NUTS, h_NUTS = sample_chain(sampler_seed, target_seed, n;
            alotted_adapt_time = adapttime_RNUTS, sampler_specs = nuts_specs, target_name, target_specs, return_chain = true)

        # coompute KS stat
        ks_prostate_nuts = []
        ks_prostate_rnuts = []
        for i in 1:(target_test.dim)
            push!(ks_prostate_nuts, HypothesisTests.ksstats(vec(chain_NRPT[:,i,1]),vec(chain_NUTS[:,i,1]))[3])
            push!(ks_prostate_rnuts, HypothesisTests.ksstats(vec(chain_NRPT[:,i,1]),vec(chain_RNUTS[:,i,1]))[3])
        end

        # store stats
        GC.gc()
        df_NUTS_KS = DataFrame(ks_prostate_nuts',:auto)
        df_NUTS_stats = DataFrame(metric = :diagonal, runtime = runtime_NUTS, adapttime = adapttime_NUTS, 
            gradient_calls = n_steps_NUTS, acceptance_rate = mean(chain_NUTS[:acceptance_rate]), 
            step_size = chain_NUTS[:step_size][end], mean_n_steps = mean(chain_NUTS[:n_steps]))
        df_NUTS = hcat(df_NUTS_KS, df_NUTS_stats)

        df_RNUTS_KS = DataFrame(ks_prostate_rnuts',:auto)
        df_RNUTS_stats = DataFrame(metric = :diag_rie, runtime = runtime_RNUTS, adapttime = adapttime_RNUTS, 
            gradient_calls = n_steps_RNUTS, acceptance_rate = mean(chain_RNUTS[:acceptance_rate]), 
            step_size = chain_RNUTS[:step_size][end], mean_n_steps = mean(chain_RNUTS[:n_steps]))
        df_RNUTS = hcat(df_RNUTS_KS, df_RNUTS_stats)

        
        isdir("${filed(arg)}") || mkdir("${filed(arg)}")
        CSV.write("${filed(arg)}/summary_NUTS.csv", df_NUTS)
        CSV.write("${filed(arg)}/summary_RNUTS.csv", df_RNUTS)

        # store chains
        GC.gc()
        df_NRPT = DataFrame(Array(chain_NRPT),:auto)
        df_NUTS = DataFrame(Array(chain_NUTS),:auto)
        df_RNUTS = DataFrame(Array(chain_RNUTS),:auto)

        CSV.write("${filed(arg)}/chain_NUTS.csv", df_NUTS)
        CSV.write("${filed(arg)}/chain_RNUTS.csv", df_RNUTS)
        CSV.write("${filed(arg)}/chain_NRPT.csv", df_NRPT)
    end

    main()
    """
}