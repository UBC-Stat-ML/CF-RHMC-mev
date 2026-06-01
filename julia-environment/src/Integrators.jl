#################################################################
# Integrator structures and methods
#################################################################

# define AdvancedHMC funtions for all integrator so no error pops up
AdvancedHMC.step_size(lf::AdvancedHMC.AbstractIntegrator) = lf.ϵ
AdvancedHMC.jitter(::Union{AbstractRNG,AbstractVector{<:AbstractRNG}}, lf::AdvancedHMC.AbstractIntegrator) = lf
function AdvancedHMC.temper(
    lf::AdvancedHMC.AbstractIntegrator, r, ::NamedTuple{(:i, :is_half),<:Tuple{Integer,Bool}}, ::Int
)
    return r
end

# add current time to stat for mixing time plots
AdvancedHMC.stat(lf::AdvancedHMC.AbstractIntegrator) = (step_size=AdvancedHMC.step_size(lf), nom_step_size=AdvancedHMC.nom_step_size(lf), current_time=time_ns()/1.0e9)
AdvancedHMC.stat(lf::AdvancedHMC.AbstractLeapfrog) = (step_size=AdvancedHMC.step_size(lf), nom_step_size=AdvancedHMC.nom_step_size(lf), current_time=time_ns()/1.0e9)


AdvancedHMC.update_nom_step_size(lf::AdvancedHMC.AbstractIntegrator, ϵ) = AdvancedHMC.@set lf.ϵ = ϵ

#################################################################
# Integrator structures
#################################################################

#### Implicit midpoint

@auto struct ImplicitMidpoint <: AdvancedHMC.AbstractIntegrator
    "Step size."
    ϵ
    
    "Maximum fixed point steps"
    n::Int

    "Fixed point error tolerance"
    δ
end

ImplicitMidpoint(ϵ) = ImplicitMidpoint(ϵ,50,0.01)

AdvancedHMC.step_size(integrator::ImplicitMidpoint) = integrator.ϵ

#################################################################
# Step function for different integrators
#################################################################

# general case
function AdvancedHMC.step(
    lf::AdvancedHMC.AbstractIntegrator,
    h::AdvancedHMC.Hamiltonian{<:AbstractRiemannianMetric},
    z::AdvancedHMC.PhasePoint,
    n_steps::Int = 1;
    fwd::Bool = n_steps > 0,  # simulate hamiltonian backward when n_steps < 0
    full_trajectory::Val{FullTraj} = Val(false),
) where {FullTraj}
    n_steps = abs(n_steps)  # to support `n_steps < 0` cases

    ϵ = fwd ? AdvancedHMC.step_size(lf) : -AdvancedHMC.step_size(lf)
    ϵ = ϵ'

    z_new = copy(z) # make a new phasepoint so that the old one won't get changed after the update

    res = if FullTraj
        Vector{AdvancedHMC.PhasePoint}(undef, n_steps)
    else
        z_new
    end

    for i = 1:n_steps
        # leap step for current integrator
        z_new = leap_step(lf,h,z_new,ϵ)
        z_temp = copy(z_new)

        # Update result
        if FullTraj
            res[i] = z_temp
        else
            res = z_temp
        end
        if !isfinite(z_temp)
            # Remove undef
            if FullTraj
                res = res[isassigned.(Ref(res), 1:n_steps)]
            end
            break
        end
    end
    return res
end


#### Leap step for any integrator

# general case
function leap_step(
    lf::AdvancedHMC.AbstractIntegrator,
    h::AdvancedHMC.Hamiltonian,
    z::AdvancedHMC.PhasePoint,
    ϵ::Float64,
)
    z = full_leap(lf,h,z,ϵ)
    return z
end
