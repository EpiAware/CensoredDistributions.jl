function _get_bijector(
        ::Type{D}, init_params::AbstractVector{<:Real}) where {D <: Distribution}
    # Create a dummy instance of the distribution to get its bijector
    dummy_dist = D(init_params...)
    return bijector(dummy_dist)
end
