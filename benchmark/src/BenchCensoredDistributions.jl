module BenchCensoredDistributions

using BenchmarkTools, Distributions
using DynamicPPL
using DynamicPPL.TestUtils
using DynamicPPL.TestUtils.AD
using Turing
using ADTypes, ForwardDiff, ReverseDiff, Mooncake

suite = BenchmarkGroup()

include("make_suite.jl")
include("primarycensored.jl")
include("discretised.jl")
include("withinintervalcensored.jl")
include("weight.jl")

end