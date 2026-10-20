include("src/sampling_functions.jl")
include("src/plotting_functions.jl")


time_record = get_summary_df("timing_comparison")

p = boxplot(log2.(time_record.dimension), log2.(time_record.time_cache ./ time_record.n_steps_cache), 
    bar_width = 0.5, label = "CF-RHMC", xlabel = "log₂(Dimension)", ylabel = "log₂(Time (s)/gradient call)")
boxplot!(log2.(time_record.dimension), log2.(time_record.time_no_cache ./ time_record.n_steps_no_cache), 
    bar_width = 0.5, label = "AdvancedHMC")
plot!((1:11) .- 17, label="linear rate", linestyle=:dash, seriescolor = :black)
plot!(2*(1:11) .- 15, label="quadratic rate", linestyle=:dot, seriescolor = :black)
plot!(3*(1:11) .- 21, label="cubic rate", linestyle=:dashdot, seriescolor = :black)

