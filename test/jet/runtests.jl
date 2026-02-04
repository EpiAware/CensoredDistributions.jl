#!/usr/bin/env julia
# JET code linting tests
# Run from the test/jet directory or with --project=test/jet

using JET
using CensoredDistributions

JET.test_package(CensoredDistributions; target_modules = (CensoredDistributions,))
