module OptimizationExt

using CensoredDistributions
using Optimization
using OptimizationOptimJL
using Bijectors
using Distributions
using Statistics

# Import product_distribution for product distribution fitting
import Distributions: product_distribution

# Import and extend fitting methods from Distributions.jl
import Distributions: fit, fit_mle

# Import the specific types and functions we need
using CensoredDistributions: IntervalCensored, PrimaryCensored,
                             interval_censored, double_interval_censored, weight,
                             NumericSolver, AnalyticalSolver

# Include the generic infrastructure
include("generic/optimization.jl")
include("generic/bijectors.jl")

# Include distribution-specific fitting methods
include("distributions/intervalcensored.jl")
include("distributions/doubleintervalcensored.jl")

end # module OptimizationExt
