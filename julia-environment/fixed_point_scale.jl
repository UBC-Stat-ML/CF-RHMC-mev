#######################################################################################
# fixed point iteration scaling experiment
#######################################################################################

include("src/sampling_functions.jl")
include("src/plotting_functions.jl")

exp_name = "fixed_point_scale"
df = get_summary_df(exp_name,"output","summary_RNUTS.csv")

df_funnel = df[df.target .== "funnel"]
df_glm = df[df.target .== "glm"]

groupedboxplot(log2.(df_glm.dimension), (df_glm.gradient_calls - 50*df_glm.integrator_fails) ./ 
(df_glm.integration_steps - df_glm.integrator_fails),
xlabel = "log₂(Dimension)", ylabel = "Gradient calls per integration step", label = "")

groupedboxplot(log2.(df_glm.dimension), 100*(df_glm.integrator_fails ./ df_glm.integration_steps),
xlabel = "log₂(Dimension)", ylabel = "Percentage of failed integration steps", label = "", ylims = [-3,100])

groupedboxplot(log2.(df_funnel.dimension), (df_funnel.gradient_calls .- 50 .* df_funnel.integrator_fails) ./ 
(df_funnel.integration_steps - df_funnel.integrator_fails),
xlabel = "log₂(Dimension)", ylabel = "Gradient calls per integration step", label = "")

groupedboxplot(log2.(df_funnel.dimension), 100 * df_funnel.integrator_fails ./ df_funnel.integration_steps,
xlabel = "log₂(Dimension)", ylabel = "Percentage of failed integration steps", label = "", ylims = [-3,100])