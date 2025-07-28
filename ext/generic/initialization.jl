"""
Parameter initialization utilities for CensoredDistributions.jl

This file provides default parameter initialization methods for different
distribution types used in fitting censored distributions.
"""

using Statistics

# Parameter initialization functions for IntervalCensored distributions
function _get_default_init_params(::Type{<:Normal}, data, interval_spec)
    continuous_approx = _interval_to_continuous_approx(data, interval_spec)
    μ_init = mean(continuous_approx)
    σ_init = std(continuous_approx)
    return [μ_init, σ_init]
end

function _get_default_init_params(::Type{<:Exponential}, data, interval_spec)
    continuous_approx = _interval_to_continuous_approx(data, interval_spec)
    θ_init = mean(continuous_approx)
    return [θ_init]
end

function _get_default_init_params(::Type{<:Gamma}, data, interval_spec)
    continuous_approx = _interval_to_continuous_approx(data, interval_spec)
    # Method of moments for Gamma(α, θ)
    m = mean(continuous_approx)
    v = var(continuous_approx)
    α_init = m^2 / v
    θ_init = v / m
    return [α_init, θ_init]
end

function _get_default_init_params(::Type{<:LogNormal}, data, interval_spec)
    continuous_approx = _interval_to_continuous_approx(data, interval_spec)
    # Remove non-positive values for LogNormal
    positive_data = continuous_approx[continuous_approx .> 0]
    if length(positive_data) == 0
        return [0.0, 1.0]  # Default LogNormal parameters
    end
    log_data = log.(positive_data)
    return [mean(log_data), std(log_data)]
end

function _get_default_init_params(::Type{<:Uniform}, data, interval_spec)
    continuous_approx = _interval_to_continuous_approx(data, interval_spec)
    # Use data range with some padding
    min_val = minimum(continuous_approx)
    max_val = maximum(continuous_approx)
    padding = (max_val - min_val) * 0.1
    return [min_val - padding, max_val + padding]
end

function _get_default_init_params(dist_type, ::Any, ::Any)
    throw(ArgumentError(
        "Default initialization not implemented for $(dist_type). " *
        "Please provide init_params."
    ))
end

# Helper functions specific to double interval censored fitting
function _get_default_init_params_double(::Type{<:LogNormal}, data, interval)
    continuous_approx = _interval_to_continuous_approx(data, interval)
    positive_data = continuous_approx[continuous_approx .> 0]
    if length(positive_data) == 0
        return [0.0, 1.0]
    end
    log_data = log.(positive_data)
    return [mean(log_data), std(log_data)]
end

function _get_default_init_params_double(::Type{<:Uniform}, data, interval)
    continuous_approx = _interval_to_continuous_approx(data, interval)
    data_range = maximum(continuous_approx) - minimum(continuous_approx)
    return [0.0, max(data_range * 0.3, 1.0)]
end

function _get_default_init_params_double(dist_type, ::Any, ::Any)
    throw(ArgumentError(
        "Default initialization not implemented for double censored " *
        "$(dist_type). Please provide initial parameters."
    ))
end

function _get_default_init_params_primary(::Type{<:Uniform}, max_pwindow::Real)
    return [0.0, max_pwindow]
end

function _get_default_init_params_primary(::Type{<:Exponential}, max_pwindow::Real)
    return [max_pwindow / 2]
end

function _get_default_init_params_primary(primary_type, max_pwindow)
    throw(ArgumentError(
        "Default initialization not implemented for primary " *
        "distribution $(primary_type). Please provide primary_init."
    ))
end