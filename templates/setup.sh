#!/bin/bash

# if insider the apptainer container, start ssh agent and add key
if [ -v APPTAINER_NAME ] ; then
    echo "initializing ssh agent and adding deploy key"
    chmod 600 $baseDir/keys/deploy_key
    eval `ssh-agent -s`
    ssh-add $baseDir/keys/deploy_key
fi

# need to do this here so that nextflow can fill in julia_env and baseDir
# also, cannot run the jl script in another process because then the container is reloaded
# so the ssh setup disappears
cat <<EOF > temp.jl
    using Pkg
    Pkg.activate("$julia_env")
    Pkg.instantiate()
    Pkg.precompile()
EOF

julia temp.jl