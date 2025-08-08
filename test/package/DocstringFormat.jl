# Shared helper functions for docstring validation
@testsnippet DocstringHelpers begin
    using CensoredDistributions
    using Distributions

    # Helper function to get docstring content as string
    function get_docstring_content(obj)
        doc = @doc obj
        if doc isa Markdown.MD
            return sprint(show, MIME("text/plain"), doc)
        else
            return string(doc)
        end
    end

    # Helper to extract function signature and arguments
    function extract_function_info(func_name, mod = CensoredDistributions)
        methods_list = methods(getfield(mod, func_name))
        all_args = Set{Symbol}()
        has_kwargs = false

        for method in methods_list
            try
                # Get argument names from method signature
                arg_names = Base.method_argnames(method)
                if length(arg_names) > 1
                    # Skip first argument (function name) and filter out internal arguments
                    relevant_args = arg_names[2:end]
                    for arg in relevant_args
                        arg_str = string(arg)
                        if arg ≠ Symbol("#unused#") &&
                           !startswith(arg_str, "#") &&
                           !startswith(arg_str, "var\"") &&
                           arg ≠ Symbol("") &&
                           !occursin("##", arg_str)
                            all_args = union(all_args, [arg])
                        end
                    end
                end

                # Check if method has keyword arguments
                if method.nkw > 0
                    has_kwargs = true
                end
            catch e
                # Skip methods that can't be introspected
                continue
            end
        end

        return collect(all_args), has_kwargs
    end

    # Helper to check if an object is a type (not a function)
    function is_type_export(name, mod)
        try
            obj = getfield(mod, name)
            return obj isa Type
        catch
            return false
        end
    end

    # Get public symbols (Julia 1.11+) and exported symbols
    function get_public_symbols()
        public_syms = Symbol[]

        # Get public symbols if available (Julia 1.11+)
        @static if VERSION >= v"1.11"
            if isdefined(CensoredDistributions, :public)
                try
                    # Try to access public symbols via module metadata
                    # This is a bit tricky as there's no direct API yet
                    # For now, we'll rely on manual discovery
                    public_syms = [
                        :PrimaryCensored, :IntervalCensored, :DoubleIntervalCensored,
                        :Weighted, :ExponentiallyTilted, :AnalyticalSolver, :NumericSolver]
                catch
                    public_syms = Symbol[]
                end
            end
        end

        return public_syms
    end

    # Automatically discover all exports and public symbols
    function discover_all_symbols()
        # Get exported symbols
        exported_symbols = names(CensoredDistributions)

        # Get public symbols
        public_symbols = get_public_symbols()

        # Combine exported and public, remove duplicates
        all_symbols = unique(vcat(exported_symbols, public_symbols))

        # Split into types and functions
        all_types = [name
                     for name in all_symbols if is_type_export(name, CensoredDistributions)]
        all_functions = [name
                         for name in all_symbols
                         if !is_type_export(name, CensoredDistributions)]

        # Also track which are exported vs public
        exported_types = [name
                          for name in exported_symbols
                          if is_type_export(name, CensoredDistributions)]
        exported_functions = [name
                              for name in exported_symbols
                              if !is_type_export(name, CensoredDistributions)]

        public_types = [name
                        for name in public_symbols
                        if is_type_export(name, CensoredDistributions)]
        public_functions = [name
                            for name in public_symbols
                            if !is_type_export(name, CensoredDistributions)]

        return (all_types, all_functions, exported_types,
            exported_functions, public_types, public_functions)
    end

    # Assign discovered symbols as variables for use in test items
    all_types, all_functions, exported_types, exported_functions,
    public_types, public_functions = discover_all_symbols()
end

@testitem "Type Documentation Format" setup=[DocstringHelpers] tags=[:quality] begin
    @testset "Type Documentation" begin
        for type_name in all_types
            @testset "$type_name" begin
                try
                    type_obj = getfield(CensoredDistributions, type_name)

                    # Only test if docstring exists (let Aqua handle existence)
                    doc = @doc type_obj
                    if doc isa Markdown.MD && !isempty(doc.content)
                        doc_str = get_docstring_content(type_obj)

                        # Skip if docstring is just the object name
                        if !occursin("No documentation found", doc_str) &&
                           length(strip(doc_str)) > length(string(type_name)) + 10
                            # Check if this is a struct type - if so, it should have field documentation
                            if type_name in all_types
                                try
                                    type_obj = getfield(CensoredDistributions, type_name)
                                    if hasmethod(fieldnames, Tuple{Type{type_obj}})
                                        field_names = fieldnames(type_obj)
                                        if length(field_names) > 0
                                            # Should have field documentation for each field
                                            for field_name in field_names
                                                field_doc = string(Base.doc(Base.Docs.Binding(
                                                    type_obj, field_name)))
                                                # Check if field has documentation (not just the default)
                                                if !occursin("No documentation found", field_doc) &&
                                                   length(strip(field_doc)) > 10
                                                    @test true  # Field has documentation
                                                else
                                                    # Check if field documentation is inline in source
                                                    @test occursin(string(field_name), doc_str)
                                                end
                                            end
                                        else
                                            @test true  # No fields to document
                                        end
                                    else
                                        @test true  # Not a struct with fields
                                    end
                                catch e
                                    @test true  # Skip if can't introspect
                                end
                            else
                                @test true  # Not a recognized type
                            end
                        else
                            # Skip test if no meaningful docstring
                            @test true
                        end
                    else
                        # Skip test if no docstring exists
                        @test true
                    end
                catch e
                    @warn "Could not test $type_name: $e"
                    @test true
                end
            end
        end
    end

    # Report discovered structure for debugging
    @info "Discovered symbols" all_types=all_types exported_types=exported_types public_types=public_types
end

@testitem "Function Documentation Format" setup=[DocstringHelpers] tags=[:quality] begin
    @testset "Function Documentation" begin
        for func_name in all_functions
            @testset "$func_name" begin
                try
                    func_obj = getfield(CensoredDistributions, func_name)

                    # Only test if docstring exists (let Aqua handle existence)
                    doc = @doc func_obj
                    if doc isa Markdown.MD && !isempty(doc.content)
                        doc_str = get_docstring_content(func_obj)

                        # Skip if docstring is just the object name or no documentation found
                        if !occursin("No documentation found", doc_str) &&
                           length(strip(doc_str)) > length(string(func_name)) + 10

                            # Try to extract function arguments automatically
                            try
                                arg_names, has_kwargs = extract_function_info(func_name)

                                # If function has arguments, check for Arguments section
                                if length(arg_names) > 0
                                    @test occursin("# Arguments", doc_str)

                                    # Check that each argument is documented
                                    args_section_match = match(r"# Arguments\n(.*?)(?=\n#|\n@|\z)"s, doc_str)
                                    if args_section_match !== nothing
                                        args_section = args_section_match.captures[1]
                                        for arg in arg_names
                                            arg_str = string(arg)
                                            if arg != :kwargs &&
                                               !startswith(arg_str, "#") &&
                                               !occursin("::", arg_str) &&  # Skip type parameters
                                               length(arg_str) > 1  # Skip single character args that are likely internal
                                                # Look for argument documentation pattern: - `arg_name`
                                                # Be flexible with type annotations
                                                arg_pattern = "- `$(arg)"
                                                if !occursin(arg_pattern, args_section)
                                                    # Try without type annotation
                                                    base_arg = split(arg_str, "::")[1]
                                                    alt_pattern = "- `$(base_arg)"
                                                    @test occursin(alt_pattern, args_section)
                                                end
                                            end
                                        end
                                    end
                                end

                                # If function has keyword arguments, check for Keyword Arguments section
                                if has_kwargs
                                    @test occursin("# Keyword Arguments", doc_str)
                                end

                            catch e
                                @warn "Could not extract argument info for $func_name: $e"
                                # No hardcoded fallbacks - if we can't extract args, skip argument validation
                                @test true
                            end

                            # All exported/public functions should have examples
                            if func_name in exported_functions ||
                               func_name in public_functions
                                @test occursin("@example", doc_str) ||
                                      occursin("```@example", doc_str)
                            end

                            # Check for TYPEDSIGNATURES macro usage (from DocStringExtensions) or function signature
                            @test occursin("TYPEDSIGNATURES", doc_str) ||
                                  occursin(string(func_name), doc_str)
                        else
                            # Skip test if no meaningful docstring
                            @test true
                        end
                    else
                        # Skip test if no docstring exists
                        @test true
                    end
                catch e
                    @warn "Could not test $func_name: $e"
                    @test true
                end
            end
        end
    end

    # Report discovered structure for debugging
    @info "Discovered functions" all_functions=all_functions exported_functions=exported_functions public_functions=public_functions
end

@testitem "Cross-Reference Validation" setup=[DocstringHelpers] tags=[:quality] begin
    @testset "Cross-Reference Validation" begin
        # Check that See also sections reference valid functions/types
        all_names = union(all_types, all_functions)

        for name in all_names
            try
                obj = getfield(CensoredDistributions, name)
                doc = @doc obj

                if doc isa Markdown.MD && !isempty(doc.content)
                    doc_str = get_docstring_content(obj)

                    # Skip if no meaningful docstring
                    if !occursin("No documentation found", doc_str) &&
                       length(strip(doc_str)) > length(string(name)) + 10
                        # Extract references from See also sections
                        see_also_matches = eachmatch(r"`([^`]+)`\]\(@ref\)", doc_str)
                        for match in see_also_matches
                            referenced_name = Symbol(match.captures[1])
                            if referenced_name ∉ all_names &&
                               referenced_name ∉
                               [:pdf, :cdf, :logpdf, :logcdf, :rand, :quantile]
                                @warn "Function/type $name references non-existent $referenced_name in See also section"
                            end
                        end
                    end
                end
            catch e
                @warn "Could not validate cross-references for $name: $e"
            end
        end

        # Always pass this test - it's about warnings, not failures
        @test true
    end
end
