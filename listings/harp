PROGRAM harp_benchmark
────────────────────────────────────────────────────────────────────────────────
INPUTS:
	kernel:          A kernel to benchmark
	implementations: A list of implementations to compare
	variants:        A list of variants for each implementation, if any
	datatype:        The datatype to use
	dimensions:      A list of dimensions for generating the datasets
	rng_seed:        A seed for a randomized data set generation
────────────────────────────────────────────────────────────────────────────────
OUTPUT:
	A list of statistics for each dimension/implementation/variant
	combination of the benchmarked kernel
────────────────────────────────────────────────────────────────────────────────
CONSTANTS:
	MIN_REP_COUNT: Minimum number of benchmarks to perform for a given
				   dimension/implementation/variant combination
	MIN_EXEC_TIME: Minimum execution time to validate a kernel benchmark
────────────────────────────────────────────────────────────────────────────────
VARIABLES:
	dataset:   A list of randomly generated values for each dimension
	samples:   A list of execution times for each
	           dimension/implementation/variant combination 
	exec_time: Execution time of a given kernel
	           dimension/implementation/variant combination
────────────────────────────────────────────────────────────────────────────────
PROCEDURE:
FOR EACH dim IN dimensions
	dataset <- generate_dataset(datatype, dim, rng_seed)

	FOR EACH impl IN implementations
		FOR EACH var IN variants
			FOR EACH i IN [0, MIN_REP_COUNT]
				WHILE exec_time < MIN_EXEC_TIME
					exec_time <- chrono(kernel(impl, var, dataset))
				END WHILE
				samples[dim, impl, var, i] <- exec_time
			END FOR EACH
		END FOR EACH
	END FOR EACH
END FOR EACH

RETURN compute_statistics(samples)
