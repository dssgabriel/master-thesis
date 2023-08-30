#show raw.where(block: true): it => {
    set text(font: "IBM Plex Mono")
    set align(left)
    set block(fill: luma(240), inset: 10pt, radius: 4pt, width: 100%)
    it
}
#show raw.where(block: false): box.with(
  fill: luma(240),
  inset: (x: 3pt, y: 0pt),
  outset: (y: 3pt),
  radius: 2pt
)
#show raw.where(block: false): text.with(font: "IBM Plex Mono")

= Contributions

This section details the work that has been conducted during the internship. We start by establishing the current state of the art for Rust's present capabilities in GPU programming. Then, we present the open-source contributions that have been made as part of the Rust-CUDA project. We continue by offering a detailed overview of a tool for profiling the performance of hardware-accelerated Rust code. Finally, we discuss the process of porting a partitioning algorithm from a CEA application on NVIDIA GPUs. 

== Establishing the state of the art

The first goal of the internship was to establish a comprehensive state of the art for programming GPUs with Rust. First, we investigate the state of the language's native support. Second, we take a look at libraries that provide capabilities for writing GPU code through shading languages or existing external frameworks. Third, we present Rust bindings to the OpenCL 3 API. Finally, we explore CUDA support specifically for NVIDIA GPUs. 

=== Native language support <native_support>

The Rust programming language officially supports NVIDIA's `nvptx64` architecture as a tier 2/*cite*/ compiler target/*cite*/. This includes support for the following :
- Writing kernels directly in Rust
- Intrinsics for retrieving a thread's unique identifier
- Synchronization primitives for block-level scheduling

However, this initial support is very limited compared to writing standard Rust. Indeed, kernels cannot depend on Rust's standard library. Kernels must be declared as `unsafe` functions, which reduces the ability of the compiler to assert the correctness of the GPU code. Moreover, one of Rust's most useful abstractions, slices, are not usable inside functions compiled for the `nvptx64` target. This forces the use of raw pointers in order to interact with memory buffers. As pointer arithmetic is forbidden in Rust, it is necessary to use the core `add` method to correctly offset a pointer's address before de-referencing it. These shortcomings result in highly verbose kernel code, which is hard to both read and write.

#figure(caption: "Minimal example for writing a native Rust DAXPY GPU kernel")[
  ```rust
// Disable access to the standard library
#![no_std]
// Remove the requirement for a main function
#![no_main]
// Enable access to the PTX ABI and its core intrinsics
#![feature(abi_ptx, core_intrinsics)]

// Import the `nvptx` namespace
use core::arch::nvptx;

// Tell the compiler to not mangle the name of the function
#[no_mangle]
// Define the function as "unsafe" and that it must follow the PTX ABI
pub unsafe extern "ptx-kernel" fn daxpy_kernel(
    x: *const f64,
    alpha: f64,
    y: *mut f64
) {
    // Compute the thread's index
    let idx = (nvptx::_block_idx_x() * nvptx::_block_dim_x()
        + nvptx::_thread_idx_x()) as usize;
    // Get a mutable borrow of the output vector's target index/address
    let item = &mut y.add(idx);
    // De-reference the target index/address to perform the AXPY operation
    *item += alpha * *&x.add(idx);
}

// Necessary code to tell the compiler what to do in case of a fatal error
#[panic_handler]
unsafe fn breakpoint_panic_handler(_: &::core::panic::PanicInfo) -> ! {
    core::intrinsics::breakpoint();
    core::hint::unreachable_unchecked();
}
  ```
]<rustc_nvptx>

As @rustc_nvptx demonstrates, a kernel as simple as DAXPY is unnecessarily verbose to write. This makes GPU code exceedingly difficult to work with in native Rust due to the high amount of complexity implied by working with no language abstractions.
#linebreak()
Furthermore, the current Rust compiler (rustc v1.72.0) is unable to produce a valid executable of the above code snippet. The CUDA runtime will throw an error stating that the provided PTX assembly (NVIDIA's proprietary high-level assembly language) is invalid when trying to load it.

There is an open tracking issue for PTX code generation problems/*cite*/ but there has not been any contribution to it since March 2022. Rust's efforts for GPU programming native support seem to currently be at a stop.

=== Compute shaders and external libraries

Shading languages seem to be the most popular approach for programming GPUs using the Rust language. There are multiple actively maintained crates that offer support for writing GPU code through compute shaders using Rust as a wrapper.
#linebreak()
The three most relevant and active libraries are the following:
- `EmbarkStudios/rust-gpu` /*cite*/
- `gfx-rs/wgpu` /*cite*/
- `vulkano-rs/vulkano` /*cite*/

Although compute shaders are a reliable way to program GPUs, they miss the point of leveraging Rust's compiler abilities to prevent a large class of parallelism-related bugs. Indeed, these libraries require the user to write the kernels using shading languages, such as GLSL/*cite*/, WGSL/*cite*/ or SPIR-V/*cite*/. Utilizing a foreign language to express GPU computations prevents the use of Rust's strict type system and unique memory management techniques to assert that the code does not contain any use-after-free, dangling pointers, or race condition kinds of bugs.

Moreover, writing scientific computing applications requires a high degree of control, especially regarding memory layout, in order to best optimize the code for a given target hardware. Most compute shaders lack this ability as they are primarily designed for graphics use cases (e.g., rendering, web interfaces, video games, etc.).

There also are external C and/or C++ libraries that provide Rust bindings, such as Arrayfire/*cite*/. Although the GPU code can be entirely written using Rust in a concise manner, these bindings are too high-level for our purpose. In the case of Arrayfire, computations are expressed using an array-based notation. This makes the code a lot more compact but means that we must rely on the library's backend code generation to do all the heavy lifting regarding optimizations.

=== OpenCL

As mentioned in @low_lvl_gpu_prog, OpenCL is a low-level GPU programming model. There are two Rust crates that provide bindings to the OpenCL 3 API: `cogciprocate/ocl`/*cite*/ and `kenba/opencl3`/*cite*/.
Both crates feature APIs that fully leverage Rust's RAII principles and concise error handling using the `?` operator.
#linebreak()
However, kernels cannot be written directly in Rust. They must be written in OpenCL C (an extension of C99) and loaded at compile-time into the Rust code, either via a macro or by directly pasting the kernel as a string into the Rust program. Similarly to the compute shaders and external libraries presented in the previous section, this prevents the Rust compiler from guaranteeing the type, memory, and thread safety of the GPU kernels. Although this appears as limiting for our purpose, i.e., using Rust to program GPUs, it means that it is easier to integrate Rust code in an existing HPC code base that uses OpenCL as their hardware-accelerator programming language (e.g., for code portability reasons).

For the rest of this subsection, we will assume the use of the `cogciprocate/ocl` crate.

#figure(
  image("../../figures/3-contributions/opencl-c_vs_rust.svg", width: 96%),
  caption: "C vs. Rust comparison of minimal code example for launching a kernel on a GPU"
)<ocl_c_vs_rs>

The `ocl` library provides all the necessary abstractions to concisely call functions from the OpenCL API. It is able to manage platforms, devices, programs and queues, kernels, memory allocations on the GPU, and data transfers between the host and the device. All of this can be expressed in highly succinct code, thanks to Rust's elegant syntax for handling errors and automatic resource deallocation. 

@ocl_c_vs_rs demonstrates how much more compact it is to write OpenCL using Rust as a "frontend", rather than C or C++. In this example, the original C code is 165 lines long, and although it correctly handles all possible errors, it only frees the allocated resources at the end of the program, which can lead to memory leaks in case of an early caused by an error. In contrast, the Rust is only 27 lines long. All the error handling and resource deallocation logic is tightly packaged through the use of the `?` operator on each `ocl` function call. If a given call returns an error, the stack is automatically unwinded to free allocated memory, before returning the error to the callee. The complete code for both OpenCL versions can be found in the @appendix, at @ocl_c and @ocl_rs respectively.   

=== CUDA <cuda>

CUDA is a low-level, C++-based, proprietary GPU programming model specifically designed for NVIDIA hardware accelerators. However, most of CUDA's internals are language-agnostic and solely work based on PTX (Parallel-Thread eXecution)/*cite*/ and/or cubin (CUDA binary) files. PTX is NVIDIA's proprietary low-level, human-readable ISA (Instruction Set Architecture). It is the penultimate state of a kernel's representation, before being lowered to SASS (Streaming ASSembler) format and turned into a cubin file. Consequently, it means that we are not bound to use C++ for writing GPU kernels and that it is possible to utilize Rust instead, as long as we are able to compile it into PTX code.
 
The `Rust-GPU/Rust-CUDA` open-source project tries to do exactly that by offering first-class CUDA programming capabilities using Rust in place of C++. It consists of a complete software stack, providing code generation targeting NVIDIA GPUs, management of the CUDA environment, and bindings to most NVIDIA libraries aimed at HPC/AI workloads. At the moment of writing, it is by far the most advanced way of programming GPUs natively in Rust.

The `Rust-CUDA` project is composed of multiple sub-projects, some of which are independent from the others. In the remainder of this section, we will present the most relevant ones for the usage of Rust-CUDA in an HPC environment.

#h(-1.8em)
*`cust`* acts as the Rust equivalent of the CUDA C++ Runtime library. It provides all the basic tools to manage the environment surrounding GPU code execution, e.g., creating streams, allocating device-side buffers, handling data transfers between CPU and GPU memory, launching kernels, etc. In order to improve control over contexts, modules, streams, and overall performance, `cust` is implemented using bindings to the CUDA Driver API. This actually comes as a requirement, as Rust-CUDA kernels that have been compiled into PTX or cubin/fatbin files must be dynamically loaded as modules at runtime, which are only supported in the Driver API. `cust` can be used independently of the other sub-projects described here and currently is the only library that can launch CUDA kernels from Rust (i.e., it is a required dependency for executing GPU code written using the Rust's compiler native support, as discussed in @native_support). Moreover, `cust` can also be used to launch kernels that have been written in CUDA C++, as long as they are in the form of a PTX or cubin/fatbin module that can be loaded at runtime, as described previously. 

#h(-1.8em)
*`cuda-std`* is the GPU-side "standard" library for writing Rust-CUDA kernels. It provides all of the usual CUDA functions and primitives (e.g., getting a thread's index, synchronizing threads at the block level, ...) and also a wide variety of low-level intrinsics for math functions, warp-level manipulations, or address-space casting. `cuda-std` also provides macros for allocating shared memory, which we can extensively use for kernel performance optimizations.

#h(-1.8em)
*`rustc-codegen-nvvm`* is a custom backend for the Rust compiler that produces PTX code, and the most crucial component of the Rust-CUDA project for enabling Rust as a first-class CUDA programming language. It leverages NVIDIA's libNVVM to offload the code generation and most of the optimization work. The NVVM IR (Intermediate Representation) is a proprietary compiler internal representation based, at the time of writing, on LLVM 7 IR. The `rustc-codegen-nvvm` module is responsible for generating valid PTX from Rust's inner Mid-level Intermediate Representation (MIR). It first lowers MIR to LLVM 7 IR, then feeds it into libNVVM before getting the final, optimized PTX. @compilation_pipeline showcases the complete compilation process for generating a cubin/fatbin from a Rust-CUDA kernel using `rustc-codegen-nvvm`.

#figure(
  image("../../figures/3-contributions/compilation_pipeline.svg", width: 96%),
  caption: "Complete compilation pipeline of a Rust-CUDA kernel"
)<compilation_pipeline>

#h(1.8em)
*`ptx-compiler`* is a small tool that allows the compilation of PTX files into cubin or fatbin files. This allows us to avoid JIT compilation of PTX upon loading it as a module when using `cust`.

#h(-1.8em)
*`cuda-builder`* is another small tool that allows us to build our Rust-CUDA kernels into PTX or cubin/fatbin files in `build.rs` scripts, thus helping to automatize the build process GPU code in Rust projects.

Unfortunately, the Rust-CUDA project has not been maintained since the end of 2021. As the Rust compiler is in constant evolution with new releases every six weeks, the current version (v.1.72.0 at the time of writing) is now incompatible with Rust-CUDA. Likewise, the latest version of CUDA is incompatible as well due to breaking changes in the NVVM library used by `rustc-codegen-nvvm`.
#linebreak()
It is important to mention that Rust-CUDA is not a project officially endorsed by Rust or NVIDIA. It purely is an open-source piece of work from a Computer Science student who does not have the time, nor the will to continue maintaining it. Moreover, from his point of view, to really take traction and benefit from more contributions, this project should be integrated upstream, directly as part of the `rustc` compiler. Ideally, it could replace the current backend implementation for PTX code generation as it is more complete and should offer better performance, thanks to the use of NVIDIA's proprietary NVVM IR. Some bindings to HPC libraries (cuBLAS, cuSPARSE, cuFFT) are not complete yet and could probably be improved by using more idiomatic Rust wrappers.

== Open-source work on the Rust-CUDA project

In order to support the latest major release of the NVIDIA CUDA Toolkit (version 12), we had to fix the breaking changes introduced between this version and the last stable version with which Rust-CUDA was compatible, CUDA 11.8. As mentioned in @cuda, the Rust-CUDA project uses a custom compiler backend, `rustc-codegen-nvvm`, which depends on NVIDIA's libNVVM. As part of the CUDA 12 Toolkit update, a new version (v2.0) of the NVVM IR specification/*cite*/ was released.
#linebreak()
NVVM 2.0 introduced the following breaking changes:
- Removed address space conversion intrinsics.
- Stricter error checking on the supported data layouts.
- Older style loop unroll pragma metadata on loop back edges is no longer supported.
- Shared variable initialization with non-undefined values is no longer supported.
CUDA 12 also drops support for Kepler and deprecates Maxwell architectures.

We #underline[#link("https://github.com/dssgabriel/Rust-CUDA")[forked Rust-CUDA]] and fixed the breaking changes presented above, as well as updated the minimum architecture requirements for using the project. We also added support for NVIDIA's newest architectures: Hopper (HPC/AI/server-focused) and Ada Lovelace (consumer-targeted). As part of this endeavor, we took the time to enhance some of the project's documentation. We also added improved code examples that leverage more advanced, previously undocumented, features of `cuda-std`. These include the use of shared memory and tiling programming techniques, applied within an optimized General Matrix Multiply (GEMM) kernel.

We are currently working on a draft Pull Request (PR) to merge these changes into the upstream Rust-CUDA so that they benefit more people and hopefully kickstart a resumption of the project's maintenance.

== Hardware-Accelerated Rust Profiling

After establishing an exhaustive state of the art for GPU programming in Rust, we choose the most relevant code generation methods and set out to benchmark their performance. To do this, we implement an open-source tool that evaluates the performance of GPU-accelerated Rust. The HARP (Hardware-Accelerate Rust Profiling) project is hosted by the #link("https://github.com/cea-hpc/HARP")[CEA-HPC] organization on GitHub.

=== Implementation details

HARP is a simple profiler for evaluating the performance of hardware-accelerated Rust code. It aims at gauging the capabilities of Rust as a first-class language for GPGPU programming, especially in the field of scientific computing.
#linebreak()
HARP can benchmark the following kernels:
- AXPY (general vector-vector addition)
- GEMM (general dense matrix-matrix multiplication)
- Reduction (sum reduction)
- Prefix Sum (sum exclusive scan)
Please note that the Rust-CUDA implementation of the scan kernel currently does not work for unknown reasons. It appears to be caused by a memory issue (segmentation fault) but the identical CUDA C++ code works flawlessly. We suppose it is caused by a problem during the Rust-CUDA code generation step, maybe ABI-related, but we were not able to find a fix at the time of writing.

#h(-1.8em)
Each of the kernels is available in several implementations:
- CPU (serial and parallel);
- OpenCL;
- CUDA, using either Rust-CUDA or CUDA C++ code.
The CPU versions of the kernels serve as a baseline to compare the speedup offered by GPUs. The GPU implementations of the GEMM kernel are available in two flavors: a naive version and an optimized one that leverages shared memory with SIMD memory loads and stores, as well as tiling techniques to more efficiently use the underlying hardware architecture. 

Profiling can be done on both single-precision and double-precision floating-point formats (see IEEE 754 norm/*cite*/). At the moment, both the reduction and scan kernels only support 32-bit signed integers. This is due to time constraints that prevented the implementation of generic versions for floating-point arithmetic, which is more intricate to set up when using advanced warp-level intrinsic.

The user must specify a kernel to benchmark, as well as a set of dimensions on which to run the measurements (vector length for AXPY, reduction and scan, matrix size for GEMM). HARP then automatically performs all the benchmarking runs and generates a CSV file containing a report of the aggregated statistics for the kernel. A report includes the following information for each dimension specified in the HARP benchmark configuration:
- The target kind (either host or device);
- The implementation variant of the kernel;
- The number of elements per dimension;
- The allocates memory size in bytes;
- The total number of FP operations.
It also includes the following metrics about the kernel:
- The minimum and maximum recorded execution time;
- The median and mean (average) recorded execution time;
- The runtime standard deviation;
- The arithmetic intensity (in FLOP/Byte);
- The memory bandwidth (in GiB/s);
- The computational performance (in GFLOP/s).

HARP also provides a Python script that generates graphs from the performance reports (using pandas and plotly libraries).

=== Benchmark methodology

In order to assert the stability and correctness of the measures, we developed a systematic approach to benchmarking the kernel implementations. @harp_algo gives a high-level overview of the algorithmic methodology used to measure the performance of Rust kernels. 

#figure(caption: "Pseudo-code of the algorithm used to benchmark kernels in HARP")[
  ```
PROGRAM harp_benchmark
──────────────────────────────────────────────────────────────────────────────────
INPUTS:
  kernel:          A kernel to benchmark
  implementations: A list of implementations to compare
  variants:        A list of variants for each implementation
  datatype:        The datatype to use
  dimensions:      A list of dimensions for generating the datasets
  rng_seed:        A seed for a randomized dataset generation
──────────────────────────────────────────────────────────────────────────────────
OUTPUT:
  A list of statistics for each dimension/implementation/variant combination
  of the benchmarked kernel
──────────────────────────────────────────────────────────────────────────────────
CONSTANTS:
  MIN_EXEC_TIME: Minimum execution time to validate a kernel execution
  MIN_REP_COUNT: Minimum number of benchmarks to perform for a given
                 dimension/implementation/variant combination
──────────────────────────────────────────────────────────────────────────────────
VARIABLES:
  dataset:   A list of randomly generated values for each dimension
  samples:   A list of execution times for each
             dimension/implementation/variant combination 
  exec_time: Execution time of a given kernel
             dimension/implementation/variant combination
──────────────────────────────────────────────────────────────────────────────────
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
  ```
]<harp_algo>

For each specified dimension to use, the input dataset is randomly initialized and remains invariant for all of the benchmark runs for that specific dimension. This guarantees that we do not fall into edge cases where the compiler or the CPU/GPU microarchitecture is able to aggressively optimize some of the computations (e.g. when doing operations with 1s or 0s). It also ensures that all implementations and their respective variants are compared using a consistent dataset.
#linebreak()
The `MIN_REP_COUNT` constant allows us to repeat the measurements as many times as necessary in order to compute meaningful statistics about the kernel's performance. The default value is set to 31 (the same value used by the MAQAO HPC profiler/*cite*/).
#linebreak()
The `MIN_EXEC_TIME` constant serves a tight loop that ensures that the kernels run for a long enough period of time. This value depends on the precision of the clock used to benchmark the kernels.

=== Results analysis

This subsection will present the results obtained both from HARP and from separate measurements designed to specifically compare GPU programming in Rust against other hardware-accelerator paradigms or libraries in the field of HPC.
#linebreak()
All performance results presented hereafter were done on a workstation with the following characteristics:
- *CPU:* Intel (Alder Lake) i5-12600H, 12 cores (4 P-cores, 8 E-cores)/16 threads \@ 4.5 GHz, 32 GB DDR5 RAM, 18 MB shared L3 cache.
- *GPU:* NVIDIA T600 (Turing), 640 CUDA cores, 4 GB GDDR6, 160 GB/s, 1.7 TFLOP/s in FP32, Driver: vXXX.XX.XX.
- *Software stack:* NVIDIA CUDA Toolkit Version 12.2, NVIDIA HPC SDK Version 23.7, NVIDIA OpenCL SDK, GCC v13.1, rustc v1.72.0 and v1.59.0
> TODO: Anything else?

TODO: Insert graphs + discuss metrics

TODO: Discuss CPU vs GPU, OpenCL vs CUDA

TODO: Insert Rust-CUDA vs. CUDA C++ graphs + discuss

TODO: Insert Rust-CUDA vs. cuBLAS + discuss

TODO: Briefly discuss Rust-CUDA vs. Kokkos + discuss (but results + hardware under NDA)

== Porting partitioning algorithms from a CEA application

The final stage of the internship involves porting parts of a real-world application to the GPU using Rust. This last step aims to push the boundaries of Rust GPGPU programming capabilities and explore the limits of the compiler's help in writing thread-safe kernels. Porting is done using the Rust-CUDA project, targeting NVIDIA GPUs.

=== `coupe`, a concurrent mesh partitioner

The application we chose to port is `coupe`/*cite*/, a modular, multi-threaded library for mesh partitioning, written in Rust. It is developed at the CEA/DAM by the joint CEA --- Paris-Saclay University LIHPC laboratory. `coupe` implements multiple algorithms aimed at achieving optimal load balancing, while also minimizing communication costs through the use of geometric methods. Hereafter, we list some of the partitioning algorithms available in the tool:
- Space-filling curves: Z-curve, Hilbert curve
- Recursive Coordinate Bisection (RCB) and Recursive Inertial Bisection (RIB)
- Multi-jagged
- Karmarkar-Karp
- K-means
Some of the algorithms offer optimized variants for cartesian meshes.

=== Recursive Coordinate Bisection (RCB)

The algorithm we have chosen to port is the Recursive Coordinate Bisection (RCB)/*cite*/. It is one of the simplest geometric algorithms.
#linebreak()
Given an N-dimensional set of points, select a vector $n$ of the canonical basis $(e_0, ..., e_(n-1))$. Split the set of points with a hyperplane orthogonal to $n$, such that the two parts of the splits are evenly weighted. Recurse as many times as necessary by reapplying the algorithm to the two parts with another normal vector in each.

> TODO: Add figure for RCB

Figure X showcases the partitioning of a set of points following the RCB algorithm. We have chosen it because of its simple approach and recursive nature, which makes it ideal to use on a GPU. 

=== Observations

In practice, trying to port the RCB algorithm was exceedingly difficult. Indeed, most of it relies on a series of basic algorithms, such as reductions or prefix sums (exclusive scan). Because Rust-CUDA does not have access to library bindings that provide those primitives GPU code blocks, this implied that we had to rewrite everything ourselves. This endeavor proved to be very time-consuming and difficult to implement in an efficient way, by hand. We encountered multiple issues with kernel code generation producing invalid PTX, which meant we had to switch to CUDA C++ implementations instead of Rust-CUDA ones. In summary, the Rust-CUDA project serves as a robust foundation for crafting "basic" GPU code using Rust. However, it is not currently equipped to handle more complex hardware-accelerated programming tasks, as it lacks many useful abstractions and/or bindings to libraries that hasten the development of optimized GPU kernels for scientific computing applications.
#linebreak()
We were not able to achieve a working Rust-CUDA port of the RCB algorithm at the time of writing. However, we are investigating a different approach that should simplify the GPU implementation, while also better exploiting the architecture of the hardware accelerator. This work will be our main focus for the remainder of the internship.
