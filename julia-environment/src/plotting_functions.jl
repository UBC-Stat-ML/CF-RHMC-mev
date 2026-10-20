using Plots, StatsPlots
using Measures
using Plots.PlotMeasures: px

##################################################################
# Utilities for glm targets
##################################################################

# KS scatter plot
function glm_KS_scatter_plot(;fn_end = ".pdf", folder = "prostate_small_10")
    # read results
    df_stats_NUTS = get_summary_df(folder,"output","summary_NUTS.csv")
    df_stats_RNUTS = get_summary_df(folder,"output","summary_RNUTS.csv")

    # extract KS values
    ks_nuts = Array(df_stats_NUTS[1,4:(end-7)])
    ks_rnuts = Array(df_stats_RNUTS[1,4:(end-7)])

    # plot results
    p = scatter(log10.(ks_nuts), log10.(ks_rnuts), label = "",
    xlims = [-2.5,0], ylims = [-2.5,0], aspect_ratio = 1,
    xlabel = "Log₁₀ KS between NUTS and NRPT",
    ylabel = "Log₁₀ KS between CF-RNUTS and NRPT",)
    scatter!(log10.(ks_nuts[57:57]), log10.(ks_rnuts[57:57]), label = "", markercolor = :white, markersize=10)
    scatter!(log10.(ks_nuts[57:57]), log10.(ks_rnuts[57:57]), label = "", markercolor = 1)
    plot!([-3,-0],[-3,-0],linestyle = :dash, label = "")
    annotate!(-0.8, -1.0, Plots.text("Advantage to CF-RNUTS", 11, rotation = 45, :green))
    annotate!(-1.0, -0.8, Plots.text("Advantage to NUTS", 11, rotation = 45, :red))

    path = joinpath(dirname(base_dir()), "deliverables", folder)
    savefig(p,joinpath(path, "KS_plot" * fn_end))
end

# get symbol for param name
function get_name_string(t::LogisticRegression{<:HierarchicalPrior}, i)
    d = Int(t.dim / 2)
    param_name_vec = ["θ","λ","τ"]
    index_vec = ["₁","₂","₃","₄","₅","₆","₇","₈","₉","₀"]
    if i == 2*d
        return param_name_vec[3]
    elseif i < d+1
        param_name = param_name_vec[1]
        digit_vec = reverse(digits(i))
        for j in digit_vec
            param_name *= index_vec[j + 10*(j==0)]
        end
        return param_name
    else
        i = i - d + 1
        param_name = param_name_vec[2]
        digit_vec = reverse(digits(i))
        for j in digit_vec
            param_name *= index_vec[j + 10*(j==0)]
        end
        return param_name
    end
end

# density pair plot
function glm_pair_plots(chain_NRPT, chain_RNUTS, chain_NUTS, target, m, k;
    fn_end = ".pdf", folder = "prostate_small_10", fsize = 12, layout_style = :vertical)
    # get axis names
    xname = get_name_string(target,m)
    yname = get_name_string(target,k)

    # get axis lims
    pl = 0.01; pu = 0.99
    lx_NRPT,ux_NRPT = quantile(chain_NRPT[:,m,1],[pl, pu])
    lx_RNUTS,ux_RNUTS = quantile(chain_RNUTS[:,m,1],[pl, pu])
    lx_NUTS,ux_NUTS = quantile(chain_NUTS[:,m,1],[pl, pu])

    ly_NRPT,uy_NRPT = quantile(chain_NRPT[:,k,1],[pl, pu])
    ly_RNUTS,uy_RNUTS = quantile(chain_RNUTS[:,k,1],[pl, pu])
    ly_NUTS,uy_NUTS = quantile(chain_NUTS[:,k,1],[pl, pu])
    
    lx = min(lx_NRPT,lx_RNUTS)
    ux = max(ux_NRPT,ux_RNUTS)
    ly = min(ly_NRPT,ly_RNUTS)
    uy = max(uy_NRPT,uy_RNUTS)

    # buffer axis
    buffer_x = (ux - lx)/5; buffer_y = (uy - ly)/5;
    lx -= buffer_x; ux += buffer_x; ly -= buffer_y; uy += buffer_y;

    # get thinning spacing
    chain_len = min(length(chain_NRPT),length(chain_RNUTS),length(chain_NUTS))
    thin_NRPT = Int(round(length(chain_NRPT)/chain_len))
    thin_RNUTS = Int(round(length(chain_RNUTS)/chain_len))
    thin_NUTS = Int(round(length(chain_NUTS)/chain_len))

    # plot results
    if layout_style == :vertical
        layout = @layout [a;b]

        d_NUTS = density(chain_NUTS[:,m,:], label = "NUTS", lw = 2, color = 3,
        xlims=[lx,ux], xtickfontsize=fsize, ytickfontsize=fsize, legendfontsize=fsize+2, xguidefontsize=fsize+5, yguidefontsize=fsize+5)
        dens_ylims = ylims(d_NUTS) # set ylims to that of NUTS
        d_NRPT = density(chain_NRPT[:,m,:], label = "NRPT", lw = 2, color = 1,
        xlims=[lx,ux],ylims=dens_ylims, xtickfontsize=fsize, ytickfontsize=fsize, legendfontsize=fsize+2, xguidefontsize=fsize+5, yguidefontsize=fsize+5)
        d_RNUTS = density(chain_RNUTS[:,m,:], label = "CF-RNUTS", lw = 2, color = 2,
        xlims=[lx,ux],ylims=dens_ylims, xtickfontsize=fsize, ytickfontsize=fsize, legendfontsize=fsize+2, xguidefontsize=fsize+5, yguidefontsize=fsize+5)

        s_NRPT = scatter(chain_NRPT[1:thin_NRPT:end,m,1],chain_NRPT[1:thin_NRPT:end,k,1],alpha = 0.15,
        xlims=[lx,ux],ylims=[ly,uy],xlabel = xname, ylabel = yname, legend = false,
        xtickfontsize=fsize, ytickfontsize=fsize, legendfontsize=fsize+2, xguidefontsize=fsize+5, yguidefontsize=fsize+5)
        
        s_RNUTS = scatter(chain_RNUTS[1:thin_RNUTS:end,m,1],chain_RNUTS[1:thin_RNUTS:end,k,1],alpha = 0.15, color = 2,
        xlims=[lx,ux],ylims=[ly,uy],xlabel = xname, ylabel = yname, legend = false,
        xtickfontsize=fsize, ytickfontsize=fsize, legendfontsize=fsize+2, xguidefontsize=fsize+5, yguidefontsize=fsize+5)

        s_NUTS = scatter(chain_NUTS[1:thin_NUTS:end,m,1],chain_NUTS[1:thin_NUTS:end,k,1],alpha = 0.15, color = 3,
        xlims=[lx,ux],ylims=[ly,uy],xlabel = xname, ylabel = yname, legend = false,
        xtickfontsize=fsize, ytickfontsize=fsize, legendfontsize=fsize+2, xguidefontsize=fsize+5, yguidefontsize=fsize+5)

        p_NRPT = plot(d_NRPT,s_NRPT; layout, size = (600,800), left_margin = 4mm, right_margin = 4mm)
        p_RNUTS = plot(d_RNUTS,s_RNUTS; layout, size = (600,800), left_margin = 4mm, right_margin = 4mm)
        p_NUTS = plot(d_NUTS,s_NUTS; layout, size = (600,800), left_margin = 4mm, right_margin = 4mm)
    else
        p_NRPT = scatter(chain_NRPT[1:thin_NRPT:end,m,1],chain_NRPT[1:thin_NRPT:end,k,1],alpha = 0.15,
        xlims=[lx,ux],ylims=[ly,uy],xlabel = xname, ylabel = yname, label = "NRPT",
        xtickfontsize=fsize, ytickfontsize=fsize, legendfontsize=fsize+2, xguidefontsize=fsize+5, yguidefontsize=fsize+5)
        
        p_RNUTS = scatter(chain_RNUTS[1:thin_RNUTS:end,m,1],chain_RNUTS[1:thin_RNUTS:end,k,1],alpha = 0.15, color = 2,
        xlims=[lx,ux],ylims=[ly,uy],xlabel = xname, ylabel = yname, label = "CF-RNUTS",
        xtickfontsize=fsize, ytickfontsize=fsize, legendfontsize=fsize+2, xguidefontsize=fsize+5, yguidefontsize=fsize+5)

        p_NUTS = scatter(chain_NUTS[1:thin_NUTS:end,m,1],chain_NUTS[1:thin_NUTS:end,k,1],alpha = 0.15, color = 3,
        xlims=[lx,ux],ylims=[ly,uy],xlabel = xname, ylabel = yname, label = "NUTS",
        xtickfontsize=fsize, ytickfontsize=fsize, legendfontsize=fsize+2, xguidefontsize=fsize+5, yguidefontsize=fsize+5)
        
    end

    path = joinpath(dirname(base_dir()), "deliverables", folder)
    savefig(p_NRPT,joinpath(path, "pairplot_NRPT_$(m)_$(k)" * fn_end))
    savefig(p_RNUTS,joinpath(path, "pairplot_RNUTS_$(m)_$(k)" * fn_end))
    savefig(p_NUTS,joinpath(path, "pairplot_NUTS_$(m)_$(k)" * fn_end))

    return p_NRPT
end

# KS box plots
function glm_scale_plots(;fn_end = ".pdf", folder = "glm_scale")
    # read results
    df_stats_NUTS = get_summary_df(folder,"output","summary_NUTS.csv")
    df_stats_RNUTS = get_summary_df(folder,"output","summary_RNUTS.csv")

    # clean results
    df_all = vcat(df_stats_NUTS,df_stats_RNUTS)
    df_all.data[df_all.data .== "prostate_small"] .= "prostate"
    df_all.metric[df_all.metric .== "diag_rie"] .= "CF-RNUTS"
    df_all.metric[df_all.metric .== "diagonal"] .= "NUTS"
    data_name_vec = []
    for i in eachindex(df_all.data)
        if df_all.data_size[i] == Inf
            data_name = df_all.data[i]* "\nfull" 
        else
            data_name = df_all.data[i]* "\n" * "$(Int(df_all.data_size[i]))"
        end
        push!(data_name_vec, data_name)
    end
    df_all = hcat(DataFrame(data_name = data_name_vec),df_all)

    # plot results
    mar = 15px    
    p_med = groupedboxplot(df_all.data_name,log10.(df_all.median_ks), group = df_all.metric, bar_width = 0.5,
    xlabel = "Dataset", ylabel = "Log₁₀ median KS", size = (650,300),
    left_margin = mar, bottom_margin = mar)

    p_mean = groupedboxplot(df_all.data_name,log10.(df_all.mean_ks), group = df_all.metric, bar_width = 0.5,
    xlabel = "Dataset", ylabel = "Log₁₀ mean KS", size = (650,300),
    left_margin = mar, bottom_margin = mar)

    p_max = groupedboxplot(df_all.data_name,log10.(df_all.max_ks), group = df_all.metric, bar_width = 0.5,
    xlabel = "Dataset", ylabel = "Log₁₀ max KS", size = (650,300),
    left_margin = mar, bottom_margin = mar)

    p_min = groupedboxplot(df_all.data_name,log10.(df_all.min_ks), group = df_all.metric, bar_width = 0.5,
    xlabel = "Dataset", ylabel = "Log₁₀ min KS", size = (650,300),
    left_margin = mar, bottom_margin = mar)

    p_runtime = groupedboxplot(df_all.data_name,log10.(df_all.runtime), group = df_all.metric, bar_width = 0.5,
    xlabel = "Dataset", ylabel = "Log₁₀ sampling time (s)", size = (650,300),
    left_margin = mar, bottom_margin = mar)

    p_time = groupedboxplot(df_all.data_name,log10.(df_all.runtime .+ df_all.adapttime), group = df_all.metric, bar_width = 0.5,
    xlabel = "Dataset", ylabel = "Log₁₀ total time (s)", size = (650,300),
    left_margin = mar, bottom_margin = mar)


    # save plots
    path = joinpath(dirname(base_dir()), "deliverables", folder)
    savefig(p_med,joinpath(path, "median_KS_plot" * fn_end))
    savefig(p_mean,joinpath(path, "mean_KS_plot" * fn_end))
    savefig(p_max,joinpath(path, "max_KS_plot" * fn_end))
    savefig(p_min,joinpath(path, "min_KS_plot" * fn_end))
    savefig(p_runtime,joinpath(path, "runtime_plot" * fn_end))
    savefig(p_time,joinpath(path, "totaltime_plot" * fn_end))
end