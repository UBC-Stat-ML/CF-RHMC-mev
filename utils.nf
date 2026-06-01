// Retrieve dimension of the dataset
// Used to estimate the mem requirement of a job before launching it
process findDataDim {
    time { 10.min }
    input:
        env JULIA_DEPOT_PATH
        path julia_env
        val arg
    output:
        stdout
    script:
    """
    #!/usr/bin/env julia

    using Pkg
    Pkg.activate(joinpath("$baseDir", "$julia_env")) 
    using MAT

    if ${arg.dim} == ${params.maxDataSize}
        dim = size(matread(joinpath("$baseDir", "data", "${arg.data}" * ".mat"))["X"], 2)
        print(dim)
    else
        print(${arg.dim})
    end
    """
}

// A back of the envelope calculation to estimate required memory
// Note: work with literal floats instead of ints to avoid int overflow 
def estimateMemoryMinESS(dim, minESS) {
    6.GB +                      // fixed cost, doubles after error
    1.GB * (                    // RAM for the chain ( O(n_iter * d) )
        dim *
        1000.0 *                 // worst case iterations/ESS
        minESS *
        5.0 *                   // safety factor for possible temp storage
        Math.pow(2.0, 3 - 30 ) 
    ) +
    1.GB * (                    // additional RAM for cov matrix stuff ( O(d^2) )
        dim * dim *
        100.0 *                   // safety factor for possible temp storage
        Math.pow(2.0, 3 - 30 )  // convert number of floats to GB (bytes per float / bytes per gigabyte) 
    )
}

//// Generate cross products that keep track of both variable names and values

def crossProduct(Map<String, List<?>> mapOfLists, boolean dryRun) {
    if (dryRun) {
        def result = [:]
        for (key in mapOfLists.keySet()) {
            def list = mapOfLists[key]
            result[key] = list.first()
        }
        return Channel.of(result)
    } else 
        crossProduct(mapOfLists)
}

def crossProduct(Map<String, List<?>> mapOfLists) {
    def keys = new ArrayList(mapOfLists.keySet())
    def list = _crossProduct(mapOfLists, keys)
    return Channel.fromList(list)
}

def _crossProduct(mapOfLists, keys) {
    if (keys.isEmpty()) {
        return [[:]]
    }
    def key = keys.remove(keys.size() - 1)
    def result = []
    for (recursiveMap : _crossProduct(mapOfLists, keys))
        for (value : mapOfLists.get(key)) {
            def copy = new LinkedHashMap(recursiveMap)
            copy[key] = value 
            result.add(copy)
        }
    return result
}


//// Collect CSVs from many execs

def collectCSVs(inputChannel) { collectCSVsProcess(inputChannel.map{toJson(it)}.toList()) }

def toJson(tuple) {
    def args = tuple[0]
    def path = tuple[1]
    return groovy.json.JsonOutput.toJson([
        path: path.toString(), 
        args: args
    ])
}

process collectCSVsProcess {
    debug true
    cache true
    scratch false
    time { 10.m * Math.pow(2, task.attempt-1) }
    //memory { 2.GB * Math.pow(2, task.attempt-1) }
    errorStrategy 'retry'
    maxRetries '4'
    input: 
        // we use file not path, as we want the json strings to be dumped into files
        file jsonObjects
    output:
        path aggregated
    publishDir { deliverables(workflow) }, mode: 'copy', overwrite: true
    """
    aggregate
    """
}


process combine_process {
    memory { 30.GB }
    time { 10.hour }
    input:
        path julia_env 
        path results
    output:
        path 'output'
    publishDir { deliverables(workflow) }, mode: 'copy', overwrite: true
    """ 
    #!/usr/bin/env julia --threads=1

    using Pkg
    Pkg.activate(joinpath("$baseDir", "$julia_env"))

    using CSV 
    using CombineCSVs

    mkdir("output")
    combine_csvs(".", "output"; comment = "#")
    """
}



////

process instantiate {
    debug true
    executor 'local'
    scratch false
    time { 2.h * Math.pow(2, task.attempt-1) }
    errorStrategy 'ignore'
    maxRetries '1'
    input:
        env JULIA_DEPOT_PATH
        path julia_env
    output:
        path julia_env
    script:
        template 'setup.sh'
}



/////

process head {
    debug true 
    executor 'local'
    input:
        path aggregated
    output:
        path aggregated
    """
    # TODO: column command is not supported by bash inside container
    for i in `ls $aggregated/*.csv`
    do
        echo \$i
        head \$i | sed 's/,,/,NA,/g' | column -t -s, 
    done
    """
}

//// Git utils

process checkGitUpdated {
    cache false 
    debug true
    executor 'local'
    input:
        val dryRun
    shell:
    if (dryRun) " " else
    '''
    cd !{projectDir}
    git remote update
    UPSTREAM=${1:-'@{u}'}
    LOCAL=$(git rev-parse @)
    REMOTE=$(git rev-parse "$UPSTREAM")
    BASE=$(git merge-base @ "$UPSTREAM")

    if [ $LOCAL = $REMOTE ]; then
        echo "Branch $(git rev-parse --abbrev-ref HEAD) up to date"
        exit 0
    elif [ $LOCAL = $BASE ]; then
        echo "Need to pull"
    elif [ $REMOTE = $BASE ]; then
        echo "Need to push"
    else
        echo "Diverged"
    fi
    git status
    '''
}

process commit {
    executor 'local'
    input:
        path token // pass a channel produced at very end to make sure we commit only at completion 
        val dryRun
    script:
    if (dryRun) " " else
    """
    cd ${projectDir}
    git add deliverables/*
    git commit -m "Auto: add deliverables for run ${workflow.runName} of ${workflow.scriptName}"
    git push
    """
}


/////

def pow(int i, int j) { java.lang.Math.pow(i, j) as Integer}

////

def deliverables(workflow) { 'deliverables/' + workflow.scriptName.replace('.nf','') }

////

def filed(config) {
    return config.collect{key, value -> "$key=$value"}.join("___") 
}
