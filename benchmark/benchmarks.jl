using BenchmarkTools
using CensoredDistributions
using Distributions

const SUITE = BenchmarkGroup()

# Test data for evaluation
const TEST_X = 2.5
const TEST_P = 0.5
const TEST_XS = collect(range(0.1, 10.0, length = 100))  # Vector for batch operations

# Include benchmark definitions
include("src/primarycensored.jl")
include("src/intervalcensored.jl")
include("src/doubleintervalcensored.jl")
