"""
Parameter transformation utilities for CensoredDistributions.jl

This file uses Bijectors.jl's predefined bijectors for common distributions.
"""

using Bijectors

# Use Bijectors.jl's automatic bijectors
function _get_bijector(::Type{D}, init_params::AbstractVector{<:Real}) where {D <: Distribution}
    # Create a dummy instance of the distribution to get its bijector
    dummy_dist = D(init_params...)
    return bijector(dummy_dist)
end