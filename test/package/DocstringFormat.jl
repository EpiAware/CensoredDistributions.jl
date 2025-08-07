@testitem "Docstring format validation" tags = [:quality] begin
    using CensoredDistributions
    using Distributions

    # Find all source files to check docstring patterns
    function find_julia_files(dir)
        files = String[]
        for (root, dirs, filenames) in walkdir(dir)
            for filename in filenames
                if endswith(filename, ".jl")
                    push!(files, joinpath(root, filename))
                end
            end
        end
        return files
    end

    src_files = find_julia_files(joinpath(pkgdir(CensoredDistributions), "src"))

    @testset "No @doc raw usage (bypasses template system)" begin
        violations = Tuple{String, Int, String}[]  # (file, line_num, line_content)

        for file in src_files
            lines = readlines(file)
            for (i, line) in enumerate(lines)
                if occursin(r"@doc\s+raw", line)
                    push!(violations, (relpath(file), i, strip(line)))
                end
            end
        end

        if !isempty(violations)
            @warn "Found @doc raw usage (bypasses template system):"
            for (file, line_num, line_content) in violations
                @warn "  $file:$line_num - $line_content"
            end
        end
        @test isempty(violations)
    end

    @testset "No template macros in docstrings (handled by template)" begin
        # Template macros should NOT appear in docstrings since they're in src/docstrings.jl
        # Exception: src/docstrings.jl itself contains the template definitions
        violations = Tuple{String, Int, String}[]  # (file, line_num, line_content)

        for file in src_files
            # Skip the docstring template file itself
            if endswith(file, "src/docstrings.jl")
                continue
            end

            lines = readlines(file)
            for (i, line) in enumerate(lines)
                # Check for template macros in docstring content
                if occursin(r"\$\(TYPEDSIGNATURES\)|\$\(TYPEDEF\)|\$\(TYPEDFIELDS\)", line)
                    push!(violations, (relpath(file), i, strip(line)))
                end
            end
        end

        if !isempty(violations)
            @warn "Found template macros in docstrings (should be in template only):"
            for (file, line_num, line_content) in violations
                @warn "  $file:$line_num - $line_content"
            end
        end
        @test isempty(violations)
    end

    @testset "Example blocks in exported constructor functions" begin
        # Find exported constructor functions (ending in _censored or starting with create/make)
        exported_symbols = names(CensoredDistributions)
        constructor_functions = []

        for s in exported_symbols
            if s != :CensoredDistributions
                try
                    obj = getfield(CensoredDistributions, s)
                    if isa(obj, Function)
                        name = string(s)
                        # Look for constructor patterns
                        if occursin("_censored", name) || startswith(name, "weight")
                            push!(constructor_functions, obj)
                        end
                    end
                catch
                    continue
                end
            end
        end

        missing_examples = String[]

        for func in constructor_functions
            # Get the rendered docstring by capturing help output
            doc_str = sprint(io -> show(io, "text/plain", (@doc func)))
            func_name = string(func)

            # Skip if no docstring
            if isempty(strip(doc_str)) ||
               occursin("No documentation found", doc_str)
                continue
            end

            # Check for @example blocks (key for user-facing functions)
            if !occursin("@example", doc_str)
                push!(missing_examples, func_name)
            end
        end

        if !isempty(missing_examples)
            @warn "Constructor functions missing @example blocks: $(join(missing_examples, ", "))"
        end
        @test isempty(missing_examples)
    end

    @testset "Argument documentation completeness" begin
        # Test key functions for argument documentation
        function check_argument_docs(func, expected_args, expected_kwargs = [])
            # Get the rendered docstring by capturing help output
            doc_str = sprint(io -> show(io, "text/plain", (@doc func)))
            func_name = string(func)

            # Skip if no docstring
            if isempty(strip(doc_str)) ||
               occursin("No documentation found", doc_str)
                return String[]
            end

            missing_arg_docs = String[]

            # Check for Arguments section
            if !isempty(expected_args) && !occursin("# Arguments", doc_str)
                push!(missing_arg_docs, "$func_name missing # Arguments section")
            end

            # Check for Keyword Arguments section
            if !isempty(expected_kwargs) && !occursin("# Keyword Arguments", doc_str)
                push!(missing_arg_docs, "$func_name missing # Keyword Arguments section")
            end

            # Check individual argument documentation
            for arg in expected_args
                arg_pattern = Regex("- `$arg`:")
                if !occursin(arg_pattern, doc_str)
                    push!(missing_arg_docs, "$func_name missing documentation for argument '$arg'")
                end
            end

            for kwarg in expected_kwargs
                kwarg_pattern = Regex("- `$kwarg`:")
                if !occursin(kwarg_pattern, doc_str)
                    push!(missing_arg_docs,
                        "$func_name missing documentation for keyword argument '$kwarg'")
                end
            end

            return missing_arg_docs
        end

        all_missing = String[]

        # Test primary_censored
        append!(all_missing,
            check_argument_docs(
                primary_censored,
                ["dist", "primary_event"],  # positional args
                ["solver", "force_numeric"]  # keyword args
            ))

        # Test interval_censored variants - check the method with boundaries
        # Get the specific method that takes boundaries vector
        boundaries_method = nothing
        for m in methods(interval_censored)
            if length(m.sig.parameters) >= 3  # (function, dist, boundaries)
                param_types = m.sig.parameters[2:end]
                if any(p -> p <: AbstractVector, param_types)
                    boundaries_method = m
                    break
                end
            end
        end

        if boundaries_method !== nothing
            # Test the boundaries variant
            doc_str = sprint(io -> show(io, "text/plain", (@doc boundaries_method)))
            if !isempty(doc_str) && !occursin("No documentation found", doc_str)
                if !occursin("- `dist`:", doc_str)
                    push!(all_missing, "interval_censored missing documentation for argument 'dist'")
                end
                if !occursin("- `boundaries`:", doc_str)
                    push!(all_missing, "interval_censored missing documentation for argument 'boundaries'")
                end
            end
        end

        # Test double_interval_censored
        append!(all_missing,
            check_argument_docs(
                double_interval_censored,
                ["dist"],  # positional arg
                ["primary_event", "lower", "upper", "interval", "force_numeric"]  # keyword args
            ))

        if !isempty(all_missing)
            @warn "Missing argument documentation:"
            for item in all_missing
                @warn "  $item"
            end
        end
        @test isempty(all_missing)
    end

    @testset "No type repetition in argument docs" begin
        # Check that argument docs don't repeat type info (since TYPEDSIGNATURES shows it)
        violations = String[]
        key_functions = [
            primary_censored, interval_censored, double_interval_censored, weight, get_dist]

        for func in key_functions
            # Get the rendered docstring by capturing help output
            doc_str = sprint(io -> show(io, "text/plain", (@doc func)))
            func_name = string(func)

            # Skip if no docstring
            if isempty(strip(doc_str)) ||
               occursin("No documentation found", doc_str)
                continue
            end

            # Look for argument lines that include type annotations
            lines = split(doc_str, '\n')
            for (i, line) in enumerate(lines)
                if occursin(r"- `\w+::[^`]+`:", line)
                    push!(violations, "$func_name line $i: $line")
                end
            end
        end

        if !isempty(violations)
            @warn "Found type annotations in argument docs (redundant with TYPEDSIGNATURES):"
            for violation in violations
                @warn "  $violation"
            end
        end
        @test isempty(violations)
    end
end
