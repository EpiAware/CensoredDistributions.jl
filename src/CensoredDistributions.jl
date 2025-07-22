module CensoredDistributions

# Non-submodule imports
using DocStringExtensions
using Distributions
using Random
using Integrals

# Exported constructors and types
export primarycensored, discretise, discretize, within_interval_censored, weight
export PrimaryCensored, Discretised, WithinIntervalCensored, Weighted

include("docstrings.jl")

include("distributions/PrimaryCensored.jl")
include("distributions/Discretised.jl")
include("distributions/WithinIntervalCensored.jl")
include("distributions/Weight.jl")

end
