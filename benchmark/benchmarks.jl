using BenchmarkTools
using CensoredDistributions
using Distributions

const SUITE = BenchmarkGroup()

# Test point for evaluation
const TEST_X = 2.5
const TEST_P = 0.5

# Include benchmark definitions
include("src/primarycensored.jl")
include("src/intervalcensored.jl")
include("src/doubleintervalcensored.jl")
