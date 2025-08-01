using CairoMakie
using Distributions
using Colors

function generate_logo()
    # Set up the figure with timeline visualization
    fig = Figure(resolution = (400, 300), backgroundcolor = :transparent)

    # Create main axis for the timeline
    ax = Axis(fig[1, 1],
        backgroundcolor = :transparent,
        xgridvisible = false, ygridvisible = false,
        leftspinevisible = false, rightspinevisible = false,
        topspinevisible = false, bottomspinevisible = false,
        xticksvisible = false, yticksvisible = false,
        xticklabelsvisible = false, yticklabelsvisible = false)

    # Timeline setup
    timeline_y = 0.5
    timeline_start = 0
    timeline_end = 10

    # Primary event window - as faded bar
    primary_start = 1
    primary_end = 3
    # True event
    true_event = 2.2

    # Delay distribution - properly starting from true event (made higher)
    delay_x = range(true_event, true_event + 8, length = 1000)
    delay_dist = Gamma(2, 0.8)
    delay_y = pdf.(delay_dist, delay_x .- true_event) .* 0.8 .+ timeline_y  # Increased from 0.4 to 0.6

    # Secondary event
    observed_event = delay_x[end]
    obs_start = observed_event - 1.5
    obs_end = observed_event + 0.5
    trunc_line_x = observed_event - 4

    # Plot delay distribution
    lines!(ax, delay_x, delay_y,
        color = RGB(0.22, 0.596, 0.149), linewidth = 3)  # Julia green
    band!(ax, delay_x, timeline_y, delay_y,
        color = (RGB(0.22, 0.596, 0.149), 0.2))

    # Primary event interval box (burnt orange, faded) - extending to top of logo
    poly!(ax,
        Point2f[(primary_start, 0.45), (primary_end, 0.45),
            (primary_end, 1.0), (primary_start, 1.0)],
        color = (RGB(0.8, 0.4, 0.1), 0.3))  # Burnt orange with transparency

    # Secondary event interval box (deep purple, faded) - extending to top of logo
    poly!(
        ax, Point2f[(obs_start, 0.45), (obs_end, 0.45),
            (obs_end, 1.0), (obs_start, 1.0)],
        color = (RGB(0.5, 0.2, 0.7), 0.3))  # Deep purple with transparency

    # Plot primary event
    scatter!(ax, [true_event], [timeline_y],
        marker = :circle, markersize = 20,
        color = RGB(0.8, 0.4, 0.1),  # Burnt orange
        strokecolor = :black, strokewidth = 1)

    # Plot observed event
    scatter!(ax, [observed_event], [timeline_y],
        marker = :circle, markersize = 20,
        color = RGB(0.5, 0.2, 0.7),  # Deep purple
        strokecolor = :black, strokewidth = 1)

    # White horizontal line inside the distribution
    vlines!(ax, [trunc_line_x],
        color = :white, linestyle = :dot, linewidth = 3)

    # Set axis limits (extended to right for whiskers)
    xlims!(ax, -0.5, 12.0)
    ylims!(ax, 0.48, 0.9)

    # Hide decorations
    hidedecorations!(ax)

    return fig
end

# Generate and save the logo
logo = generate_logo()
save("docs/src/assets/logo.svg", logo)
save("docs/src/assets/logo.png", logo, px_per_unit = 4)

println("Logo generated successfully!")
println("- logo.svg and logo.png")
