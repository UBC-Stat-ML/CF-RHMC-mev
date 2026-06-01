include("../src/sampling_functions.jl")

using HypothesisTests: HypothesisTest, ApproximateTwoSampleKSTest, pvalue

### Invariance test based on Pigeons.jl

function generate_forward(true_sample,target,rng; sampler_seed = 1, sampler_specs = (;), forward_steps = 2)
    if true_sample # reference sample
        if target_name == :funnel # use exact sample
            x1_true = rand(rng,Normal(0,3))
            xm1_true = rand(rng,Normal(0,exp(x1_true/target.b)),target.dim-1)
            return vcat(x1_true,xm1_true)
        elseif target_name == :glm # use NUTS to get close to exact sample
            chain, _, _, _ = sample_hmc(target,2000, sampler_seed, 1000;metric_name = :diagonal,)
            return vec(Array(chain[end,:,1]))
        end
    else # forward sample with sampler
        chain, _, _, _ = sample_hmc(target, forward_steps, sampler_seed; sampler_specs...)
        return vec(Array(chain[end,:,1]))
    end
end

function invariance_test(
    target_seed,
    target_name,
    target_specs,
    sampler_specs, 
    ::Type{two_sample_hypothesis_test} = ApproximateTwoSampleKSTest;
    forward_steps = 2,
    n_iid_samples::Integer = 10_000,
    marginal_pvalue_threshold::Real = 0.005,
    ) where {two_sample_hypothesis_test <: HypothesisTest}
    
    # allocate storage for the initial and final samples
    initial_values = Vector{Vector{typeof(marginal_pvalue_threshold)}}(undef, n_iid_samples)
    final_values = similar(initial_values)

    # iterate iid samples
    rng = SplittableRandom(1)
    target = build_target(target_name, target_seed; target_specs...)
    for n in eachindex(initial_values)
        initial_values[n] = generate_forward(true,target,rng; sampler_seed = n)
        sampler_specs = (;sampler_specs...,initstate = initial_values[n])
        final_values[n] = generate_forward(false,target,rng; sampler_seed = n, sampler_specs, forward_steps)
    end

    # transform vector of vectors to matrices so that iterating dimensions == iterating columns => faster
    inits_mat = collect(hcat(initial_values...)')
    finals_mat = collect(hcat(final_values...)')
    @assert size(inits_mat) == size(finals_mat) "Initial values and final values have different dimensions"

    # use a Bonferroni correction for multiple testing
    corrected_pvalue_threshold = marginal_pvalue_threshold/size(finals_mat,2)

    # compute pvalues for all dimensions
    pvalues = map(zip(eachcol(inits_mat), eachcol(finals_mat))) do (x,y)
        pvalue(two_sample_hypothesis_test(x, y))
    end

    # inspect results
    failed_tests = findall(<(corrected_pvalue_threshold), pvalues)
    passed = length(failed_tests) == 0

    return (; passed, pvalues, failed_tests)
end

# glm target
target_seed = 123
target_name = :glm
target_specs = (;dim = 2, n = 3, prior_type = :normal, synthetic = true)
target = build_target(target_name, target_seed; target_specs...)


sampler_specs = (;metric_name = :diag_rie, a = Inf, step_size = 0.1)
test_res = invariance_test(target_seed, target_name, target_specs, sampler_specs; n_iid_samples = 100, forward_steps = 10)

@assert test_res[1]


# funnel target
target_seed = 123
target_name = :funnel
target_specs = (;dim = 2, b = 2.0)
target = build_target(target_name, target_seed; target_specs...)


sampler_specs = (;metric_name = :diag_rie, a = Inf, step_size = 0.1)
test_res = invariance_test(target_seed, target_name, target_specs, sampler_specs; n_iid_samples = 100)

@assert test_res[1]