#show raw.where(block: true): it => {
    set text(font: "Intel One Mono", size: 8pt)
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
#show raw.where(block: false): text.with(font: "Intel One Mono")

= Contributions

#h(1.8em)
This section details the work that has been conducted during the internship. We start by establishing the current state of the art for Rust's present capabilities in GPU programming. Then, we present the open-source contributions that have been made as part of the Rust-CUDA project. We continue by offering a detailed overview of a tool for profiling the performance of hardware-accelerated Rust code. Finally, we discuss the process of porting a partitioning algorithm from a CEA application on NVIDIA GPUs. 

== Establishing the state of the art

#h(1.8em)
The first goal of the internship was to establish a comprehensive state of the art for programming GPUs with Rust. First, we investigate the state of the language's native support. Second, we look at libraries that provide capabilities for writing GPU code through shading languages or existing external frameworks. Third, we present Rust bindings to the OpenCL 3 API. Finally, we explore CUDA support specifically for NVIDIA GPUs. 

=== Native language support <native_support>

#h(1.8em)
The Rust programming language officially supports NVIDIA's `nvptx64` architecture as a "tier 2"  compiler target @noauthor_target_nodate @noauthor_nvptx64-nvidia-cuda_nodate. This includes support for the following :
- Writing kernels directly in Rust
- Intrinsics for retrieving a thread's unique identifier
- Synchronization primitives for block-level scheduling

However, this initial support is minimal compared to writing standard Rust. Indeed, kernels cannot depend on Rust's standard library. Kernels must be declared as `unsafe` functions, which reduces the compiler's ability to assert the GPU code's correctness. Moreover, one of Rust's most useful abstractions, slices, are not usable inside functions compiled for the `nvptx64` target. This forces the use of raw pointers to interact with memory buffers. As pointer arithmetic is forbidden in Rust, it is necessary to use the core `add` method to correctly offset a pointer's address before de-referencing it. These shortcomings result in highly verbose kernel code, which is hard to read and write.

#figure(caption: "Minimal example for writing a native Rust DAXPY GPU kernel")[
  ```rust
#![no_std]  // Disable access to the standard library
#![no_main] // Remove the requirement for a main function
#![feature(abi_ptx, core_intrinsics)] // Enable PTX ABI and access to its intrinsics
use core::arch::nvptx; // Import the `nvptx` namespace

#[no_mangle] // Prevent the compiler from mangling the function's name
// Define the function as "unsafe" and use the PTX ABI
pub unsafe extern "ptx-kernel" fn daxpy_kernel(
    n: usize, alpha: f64, x: *const f64, y: *mut f64
) -> {
    let idx = nvptx::_thread_idx_x() as usize; // Retrieve the thread's index
    if idx < n { // Assert that the index is not out of bounds
        // Get a mutable borrow of the output vector's target index/address
        let item = &mut *y.add(idx);
        // De-reference the target index/address to perform the AXPY operation
        *item += alpha * &*x.add(idx);
    }
}
// Necessary code to tell the compiler what to do in case of a fatal error
#[panic_handler]
unsafe fn breakpoint_panic_handler(_: &::core::panic::PanicInfo) -> ! {
    core::intrinsics::breakpoint();
    core::hint::unreachable_unchecked();
}
  ```
]<rustc_nvptx>

#h(1.8em)
As @rustc_nvptx demonstrates, a kernel as simple as DAXPY is unnecessarily verbose to write. This makes GPU code exceedingly challenging to work with in native Rust due to the high amount of complexity implied by working with no language abstractions.
#linebreak()
Furthermore, the current Rust compiler (rustc v1.72.0) cannot produce a valid executable of the above code snippet. The CUDA runtime will throw an error stating that the provided PTX assembly (NVIDIA's proprietary high-level assembly language) is invalid when trying to load it.

There is an open tracking issue for PTX code generation problems @noauthor_nvptx_nodate, but there has not been any contribution since March 2022. Rust's efforts for GPU programming native support seem to be at a stop currently.

We did not retain Rust's native support for GPU programming as a suitable approach, as it is currently unusable. Consequently, we did not conduct any performance evaluation with it.  

=== Compute shaders and external libraries

#h(1.8em)
Shading languages are the most popular approach for programming GPUs using the Rust language. Multiple actively maintained crates offer support for writing GPU code through compute shaders using Rust as a wrapper.
#linebreak()
The three most relevant and active libraries are the following:
- `EmbarkStudios/rust-gpu` @noauthor_rust_gpu_2023
- `gfx-rs/wgpu` @noauthor_webgpu_nodate
- `vulkano-rs/vulkano` @noauthor_vulkano_2023

#h(1.8em)
Although compute shaders are a reliable way to program GPUs, they miss the point of leveraging Rust's compiler abilities to prevent a large class of parallelism-related bugs. Indeed, these libraries require the user to write the kernels using shading languages, such as GLSL/WGSL @noauthor_khronosgroupglslang_nodate or SPIR-V @noauthor_spir_2014. Utilizing a foreign language to express GPU computations prevents using Rust's strict type system and unique memory management techniques to assert that the code does not contain any use-after-free, dangling pointers or race condition kinds of bugs.

Moreover, writing scientific computing applications requires a high degree of control, especially regarding memory layout, to best optimize the code for a given target hardware. Most compute shaders lack this ability as they are primarily designed for graphics use cases (e.g., rendering, web interfaces, video games, etc.).

External C and C++ libraries, such as Arrayfire @noauthor_arrayfire_nodate, also provide Rust bindings. Although the GPU code can be entirely written concisely using Rust, these bindings are too high-level for our purpose. In the case of Arrayfire, computations are expressed using an array-based notation. This makes the code much more compact but means we must rely on the library's backend code generation to do all the heavy lifting regarding optimizations.

While compute shaders are the most popular way of programming GPUs in Rust, they do not align with the uncompromising demands of HPC and scientific computing. Consequently, we did not consider benchmarking their performance, deeming them an impractical approach for our purposes.

=== OpenCL

#h(1.8em)
As mentioned in @low_lvl_gpu_prog, OpenCL is a low-level GPU programming model. Two Rust crates provide bindings to the OpenCL 3 API: `cogciprocate/ocl` and `kenba/opencl3`.
Both crates feature APIs that fully leverage Rust's RAII principles and concise error handling using the `?` operator. However, kernels cannot be written directly in Rust. They must be written in OpenCL C (an extension of C99) and loaded at compile-time into the Rust code, either via a macro or by directly pasting the kernel as a string into the Rust program. Similarly to the compute shaders and external libraries presented in the previous section, this prevents the Rust compiler from guaranteeing GPU kernels' type, memory, and thread safety. Although this appears limiting for our purpose,  it is easier to integrate Rust code in an existing HPC code base that uses OpenCL as their hardware-accelerator programming language (e.g., for code portability reasons).

#h(-1.8em)
In the rest of this subsection, we will assume the use of the `cogciprocate/ocl` crate  @noauthor_ocl_2023.

The `ocl` library provides all the necessary abstractions to call functions from the OpenCL API concisely. It can manage platforms, devices, programs and queues, kernels, memory allocations on the GPU, and data transfers between the host and the device. This can be expressed in highly succinct code, thanks to Rust's elegant syntax for handling errors and automatic resource deallocation. 

#v(1em)
#figure(
  image("../../figures/3-contributions/opencl-c_vs_rust.svg", width: 96%),
  caption: "C vs. Rust comparison of minimal code example for launching a kernel on a GPU."
)<ocl_c_vs_rs>
#v(1em)

#h(1.8em)
@ocl_c_vs_rs demonstrates how much more compact it is to write OpenCL using Rust as a "frontend" rather than C or C++. In this example, the original C code is 165 lines long. Although it correctly handles all possible errors, it only frees the allocated resources at the end of the program, which can lead to memory leaks in case of an early caused by an error. In contrast, the Rust is only 27 lines long. All the error handling and resource deallocation logic is tightly packaged through the use of the `?` operator on each `ocl` function call. If a given call returns an error, the stack is automatically unwinded to free allocated memory before returning the error to the callee. The complete code for both OpenCL versions can be found in the @appendix at @ocl_c and @ocl_rs, respectively.

Since OpenCL already occupies an essential role in hardware-accelerator programming for HPC, owing to its emphasis on fine-grained control and cross-vendor portability, we selected it as a viable option for Rust-based GPU programming. As such, we conducted performance evaluations on OpenCL as part of the later stages of the internship, which we present in a subsequent section of this report.

=== CUDA <cuda>

#h(1.8em)
As introduced in @low_lvl_gpu_prog, CUDA is a low-level, proprietary GPU programming model designed specifically for NVIDIA hardware accelerators. While CUDA is C++-based, most of its internals are language-agnostic and solely work based on PTX (Parallel-Thread eXecution) @noauthor_ptx_nodate and/or cubin (CUDA binary) files. PTX is NVIDIA's proprietary low-level, human-readable ISA (Instruction Set Architecture), and the penultimate state of a kernel's representation before being lowered to SASS (Streaming ASSembler) format and ultimately turned into a cubin file. Consequently, this means that we are not bound to use C++ for writing GPU kernels and that it is possible to utilize Rust instead, as long as we are able to compile it into PTX code.

#pagebreak()
#h(1.8em)
The `Rust-GPU/Rust-CUDA` open-source project @noauthor_rust-gpurust-cuda_nodate tries to do that by offering Rust first-class CUDA programming capabilities instead of C++. It consists of a complete software stack, providing code generation targeting NVIDIA GPUs, management of the CUDA environment, and bindings to most NVIDIA libraries aimed at HPC/AI workloads. Although it is limited to CUDA hardware, the Rust-CUDA project is, at the moment of writing, the most advanced way of natively programming GPUs in Rust.

The `Rust-CUDA` project comprises multiple sub-projects, some of which are independent from the others. In the remainder of this section, we will present the most relevant ones for using Rust-CUDA in an HPC environment.

#h(-1.8em)
*`cust`* acts as the Rust equivalent of the CUDA C++ Runtime library. It provides all the basic tools to manage the environment surrounding GPU code execution, e.g., creating streams, allocating device-side buffers, handling data transfers between CPU and GPU memory, launching kernels, etc. In order to improve control over contexts, modules, streams, and overall performance, `cust` is implemented using bindings to the CUDA Driver API. This actually comes as a requirement, as Rust-CUDA kernels that have been compiled into PTX or cubin/fatbin files must be dynamically loaded as modules at runtime, which are only supported in the Driver API. `cust` can be used independently of the other sub-projects described here and currently is the only library that can launch CUDA kernels from Rust (i.e., it is a required dependency for executing GPU code written using the Rust's compiler native support, as discussed in @native_support). Moreover, `cust` can also be used to launch kernels written in CUDA C++, as long as they are a PTX or cubin/fatbin module that can be loaded at runtime, as described previously. 

#h(-1.8em)
*`cuda-std`* is the GPU-side "standard" library for writing Rust-CUDA kernels. It provides all of the usual CUDA functions and primitives (e.g., getting a thread's index, synchronizing threads at the block level, etc.) and also a wide variety of low-level intrinsics for math functions, warp-level manipulations, or address-space casting. `cuda-std` also provides macros for allocating shared memory, which we can extensively use for kernel performance optimizations.

#h(-1.8em)
*`rustc-codegen-nvvm`* is a custom backend for the Rust compiler that produces PTX code and the most crucial component of the Rust-CUDA project for enabling Rust as a first-class CUDA programming language. It leverages NVIDIA's libNVVM to offload the code generation and most of the optimization work. The NVVM IR (Intermediate Representation) is a proprietary compiler internal representation based, at the time of writing, on LLVM 7 IR. The `rustc-codegen-nvvm` module is responsible for generating valid PTX from Rust's inner Mid-level Intermediate Representation (MIR). It first lowers MIR to LLVM 7 IR, then feeds it into libNVVM before getting the final, optimized PTX. @compilation_pipeline showcases the complete compilation process for generating a cubin/fatbin from a Rust-CUDA kernel using `rustc-codegen-nvvm`.

#v(1em)
#figure(
  image("../../figures/3-contributions/compilation_pipeline.svg", width: 96%),
  caption: "Complete compilation pipeline of a Rust-CUDA kernel"
)<compilation_pipeline>

#pagebreak()
*`ptx-compiler`* is a small tool that allows the compilation of PTX files into cubin or fatbin files. This allows us to avoid JIT compilation of PTX upon loading it as a module when using `cust`.

#h(-1.8em)
*`cuda-builder`* is another small tool that allows us to build our Rust-CUDA kernels into PTX or cubin/fatbin files in `build.rs` scripts, thus helping to automatize the build process GPU code in Rust projects.

Unfortunately, the Rust-CUDA project has not been maintained since the end of 2021. As the Rust compiler is constantly evolving, with new releases every six weeks, the current version (v.1.72.0 at the time of writing) is now incompatible with Rust-CUDA. Likewise, the latest version of CUDA is also incompatible due to breaking changes in the NVVM library used by `rustc-codegen-nvvm`. However, this has been mitigated as part of the internship, as explained in @oss_work.

It is essential to mention that Rust-CUDA is not a project officially endorsed by Rust or NVIDIA. It is purely an open-source piece of work developed by a computer science student who does not have the time nor the will to continue maintaining it. Moreover, from his point of view, this project should be integrated upstream, directly as part of the `rustc` compiler, to really gain traction and benefit from more contributions. Ideally, it could replace the current backend implementation for PTX code generation as it is more complete and should offer better performance, thanks to the use of NVIDIA's proprietary NVVM IR. Some bindings to HPC libraries (cuBLAS, cuSPARSE, cuFFT) are incomplete and could probably be improved by using more idiomatic Rust wrappers.

While Rust-CUDA only aims at GPU programming on NVIDIA hardware, it is the advanced and complete way of writing kernels using native Rust syntax. As a result, we selected it for subsequent benchmarking experiments carried out later in this internship.

== Open-source work on the Rust-CUDA project <oss_work>

#h(1.8em)
To support the latest major release of the NVIDIA CUDA Toolkit (version 12), we had to fix the breaking changes introduced between this version and the last stable version with which Rust-CUDA was compatible, CUDA 11.8. As mentioned in @cuda, the Rust-CUDA project uses a custom compiler backend, `rustc-codegen-nvvm`, which depends on NVIDIA's libNVVM. As part of the CUDA 12 Toolkit update, a new version (v2.0) of the NVVM IR specification @noauthor_nvvm_nodate was released.

#h(-1.8em)
NVVM 2.0 introduced the following breaking changes:
- Removed address space conversion intrinsics.
- Stricter error checking on the supported data layouts.
- Older style loop unroll pragma metadata on loop back edges is no longer supported.
- Shared variable initialization with non-undefined values is no longer supported.
CUDA 12 also adds support for Hopper and Ada Lovelace architecture while dropping support for Kepler and deprecating Maxwell architectures.

We #underline[#link("https://github.com/dssgabriel/Rust-CUDA")[forked Rust-CUDA]] and fixed the breaking changes presented above, as well as updated the minimum architecture requirements for using the project. We also added support for NVIDIA's newest architectures: Hopper (HPC/AI/server-focused) and Ada Lovelace (consumer-targeted). As part of this endeavor, we took the time to enhance some of the project's documentation. We also added improved code examples that leverage more advanced, previously undocumented, features of `cuda-std`. These include using shared memory and tiling programming techniques applied within an optimized General Matrix Multiply (GEMM) kernel.

We are currently working on a draft Pull Request (PR) to merge these changes into the upstream Rust-CUDA to benefit more people and hopefully kickstart a resumption of the project's maintenance.

#pagebreak()
== Hardware-Accelerated Rust Profiling

#h(1.8em)
After establishing an exhaustive state of the art for GPU programming in Rust, we chose the most relevant code generation methods and set out to benchmark their performance. To do this, we implemented an open-source tool that evaluates the performance of GPU-accelerated Rust. HARP (Hardware-Accelerate Rust Profiling) is a CEA project hosted at the #link("https://github.com/cea-hpc/HARP")[CEA-HPC] organization on GitHub.

=== Implementation details

#h(1.8em)
HARP is a simple profiler for evaluating the performance of hardware-accelerated Rust code. It aims at gauging the capabilities of Rust as a first-class language for GPGPU programming, especially in the field of scientific computing.

#h(-1.8em)
HARP can benchmark and profile the following set of kernels:
- AXPY (general vector-vector addition), of complexity $cal(O)(n)$
  $ y = alpha x + y $
- GEMM (general dense matrix-matrix multiplication), of complexity $cal(O)(n^3)$
  $ C = alpha A B + beta C $
- Reduction (sum reduction), of complexity $cal(O)(log n)$ in parallel, $cal(O)(n)$ otherwise
  $ r = sum_(i=0)^n x_i $
- Prefix Sum (sum exclusive scan), of complexity $cal(O)(log n)$ if the processor (CPU or GPU) has at least $n$ cores, $cal(O)(n log n)$ otherwise
  $ forall x_i in x, x_i = sum_(j=0)^(i-1) x_j $

#v(1em)
#h(1.8em)
Please note that the Rust-CUDA implementation of the scan kernel currently does not work for unknown reasons. It appears to be caused by a memory issue (segmentation fault), but the identical CUDA C++ code works flawlessly. We suppose it is caused by a problem during the Rust-CUDA code generation step, maybe ABI-related, but we could not find a fix at the time of writing.

The AXPY and GEMM kernels were chosen because they are part of the traditional Basic Linear Algebra Software (BLAS) @noauthor_blas_nodate routines. Measuring the performance of BLAS kernels is crucial in HPC applications as it enables the optimization of fundamental mathematical operations, which are prevalent in scientific computing workloads. BLAS performance benchmarking helps identify bottlenecks, improve computational efficiency, and optimize hardware utilization, particularly on specialized architectures such as GPUs and multi-core CPUs. Ultimately, they aid in algorithm selection, benchmarking HPC systems, assessing scalability, and achieving energy-efficient computations. We have paid the utmost attention to optimizing the GPU-based GEMM kernels, using a myriad of advanced optimization techniques, such as shared memory, tiling, and instruction-level parallelism, based on the ClBlast OpenCL BLAS library @nugteren_clblast_2023.
#linebreak()
The reduction and scan kernels are fundamental building blocks for all sorts of algorithms and were needed to implement the RCB algorithm presented in @rcb. We took this opportunity to include them in the set of benchmarks provided by HARP. Similarly to what we have done for the BLAS kernels, we took care of optimizing their implementation using state-of-the-art GPGPU programming techniques @noauthor_chapter_nodate @noauthor_faster_nodate @harris_optimizing_nodate. 

#h(-1.8em)
Each of the kernels is available in several implementations:
- CPU: sequential naive (using C-style `for` loops), sequential idiomatic (using iterators constructs), and parallel (using the `rayon` crate);
- OpenCL;
- CUDA, using either Rust-CUDA or CUDA C++ code.

#h(1.8em)
The CPU versions of the kernels serve as a baseline to compare the speedup GPUs offer. The GPU implementations of the GEMM kernel are available in two flavors: a naive version and an optimized one that leverages shared memory with SIMD memory loads and stores, as well as tiling techniques to use the underlying hardware architecture more efficiently. 

Profiling can be done on both single-precision and double-precision floating-point formats (following IEEE 754 norm @8766229). Currently, both the reduction and scan kernels only support 32-bit signed integers. This is due to time constraints that prevented the implementation of generic versions for floating-point arithmetic, which is more intricate to set up when using advanced warp-level intrinsic. The algorithmic results are validated with an accuracy requirement: a tolerance of $10^(-15)$ for double-precision implementations and $10^(-6)$ for single-precision counterparts.

The user must specify a kernel to benchmark and a set of dimensions on which to run the measurements (vector length for AXPY, reduction and scan, matrix size for GEMM). HARP then automatically performs all the benchmarking runs and generates a CSV file containing a report of the aggregated statistics for the kernel. A report includes the following information for each dimension specified in the HARP benchmark configuration:
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

#h(1.8em)
HARP also provides a Python script that produces graphs from the performance reports using pandas and plotly libraries. It takes the CSV output from HARP and can be configured to output PNG images of the generated plots.

=== Benchmark methodology

#h(1.8em)
In order to assert the stability and correctness of the measures, we developed a systematic approach to benchmarking the kernel implementations. @harp_algo gives a high-level overview of the algorithmic methodology used to measure the performance of Rust kernels. 

The input dataset is randomly initialized for each specified dimension and remains invariant for all of the benchmark runs for that specific dimension. This guarantees that we do not fall into edge cases where the compiler or the CPU/GPU microarchitecture can aggressively optimize some computations (e.g., when doing operations with 1s or 0s). It also ensures that all implementations and their respective variants are compared using a consistent dataset.

The `MIN_REP_COUNT` constant allows us to repeat the measurements as many times as necessary to compute meaningful statistics about the kernel's performance. The default value is set to 31 (the same value used by the MAQAO HPC profiler @noauthor_maqao_nodate).

The `MIN_EXEC_TIME` constant serves a tight loop that ensures that the kernels run for a long enough period of time. This value depends on the clock's precision used to benchmark the kernels.

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

=== Results analysis

#h(1.8em)
This subsection will present the results obtained from HARP and separate measurements designed to specifically compare GPU programming in Rust against other hardware-accelerator paradigms or libraries in the field of HPC.
#linebreak()
All performance results presented hereafter were done on a workstation with the following characteristics:
- *CPU:* Intel (Alder Lake) i5-12600H, 12 cores (4 hyper-threaded P-cores, 8 E-cores) \@ 4.5 GHz, 32 GB DDR5 RAM.
- *GPU:* NVIDIA T600 (Turing), 640 CUDA cores, 4 GB GDDR6 VRAM, 160 GB/s memory bandwith, 1.7 TFLOP/s computational performance in FP32.
- *Software stack:* NVIDIA GPU Driver v535.86.10, NVIDIA CUDA Toolkit Version 12.2, NVIDIA OpenCL SDK Version 12.2, NVIDIA HPC SDK Version 23.7.
- *Compilers*: `gcc` v11.4 and v13.1, `rustc` v1.59.0 and v1.72.0, `nvcc` v12.2.

#h(1.8em)
In this subsection, we will focus on presenting results for the DGEMM kernel, which is the most relevant one in the context of HPC. In @anx_figs, we include the full plot outputs for all kernels available in HARP.

#figure(
  image("../../figures/3-contributions/dgemm_avg_runtime.png", width: 78%),
  caption: "Average runtime performance for the DGEMM kernel"
)<dgemm_avg_rtm>

#h(1.8em)
@dgemm_avg_rtm compares the average runtime of multiple Rust implementations of the DGEMM kernel (CPU and GPU) for increasing sizes of FP64 precision dense matrices. The graph includes error bars for each measurement obtained following the algorithm described in @harp_algo. This graph clearly shows the performance dominance of hardware accelerators over traditional CPUs. Both sequential implementations (naive uses C-style `for` loops, the other uses Rust's iterator constructs) are at least twice as slow as the fastest GPU version on matrices that are twice as small. Both CUDA-based DGEMMs significantly outshine the OpenCL implementations.

@dgemm_flops presents the computational performance for each kernel implementation using the same results as @dgemm_avg_rtm. We can interpret this graph as the opposite of the previous one, with higher FLOP/s indicating increased performance. This plot gives a better visualization of the performance difference between implementations, with CUDA-based ones clearly ahead of OpenCL, with over 2x better performance. We can also notice that the GPU kernels are compute-bound; their performance continually increases until it reaches a plateau and stays constant regardless of the size of the matrix. On the other hand, the results of CPU implementations decrease as the matrixes get bigger, highlighting the fact that these kernels are memory-bound.

#figure(
  image("../../figures/3-contributions/dgemm_flops.png", width: 78%),
  caption: "Computational performance for the DGEMM kernel"
)<dgemm_flops>

#grid(
  rows: 2,
  row-gutter: 10pt
)[
  #figure(
    image("../../figures/3-contributions/dgemm_perf_comp.svg", width: 80%),
    caption: "Performance comparison between DGEMM kernels on different CUDA-based implementations",
  )<dgemm>
  #figure(
    image("../../figures/3-contributions/sgemm_perf_comp.svg", width: 80%),
    caption: "Performance comparison between DGEMM kernels on different CUDA-based implementations"
  )<sgemm>
]

#h(1.8em)
@dgemm and @sgemm compare the computational performance of Rust-CUDA, CUDA C++ and cuBLAS implementations of the GEMM kernel in both single and double FP precision on an NVIDIA T600 GPU. The results presented are for matrices of size 2048, initialized with random values between 0 and 1 and non-null and non-one values for the $alpha$ and $beta$ coefficients. The same benchmarking methodology used in HARP has been applied here, and the standard deviation of these results is under 5%. 
In double-precision floating-point, Rust-CUDA kernels performed slightly better than the CUDA C++ implementation (a 1:1 equivalent). Both of these manual implementations are behind the NVIDIA cuBLAS one by about 10%, which is a relatively small performance drop considering how highly optimized NVIDIA's libraries are. As both the Rust-CUDA and CUDA C++ kernels share the same code generation pipeline (NVVM IR $arrow$ PTX, see @compilation_pipeline), @dgemm demonstrates that the Rust compiler front-end can match, and even slightly edge, the C++ one in terms of optimizations made at the IR level.

@sgemm highlights this even better, with the Rust-CUDA implementation achieving a massive 75% performance improvement over the CUDA C++ SGEMM kernel. However, this result seems overly pessimistic of the FLOP/s we expect from a CUDA C++ implementation. Historically, C and C++ compilers used to convert every floating-point operation to double precision, even if only single precision was required. Some of the arithmetic operations in the CUDA C++ implementation of the SGEMM kernel may be performed in FP64, which would explain the reduced performance. At the time of writing, we could not assert that this is what is actually happening. We are investigating at the binary level by analyzing the generated assembly (PTX and SASS) to confirm it.

On the other hand, @sgemm shows that the cuBLAS implementation is largely out of reach when dealing with single-precision floating-point arithmetic. This is explained by the cuBLAS implementation using the GPU's hardware more efficiently than our hand-written kernels, notably through extensive reliance on tensor cores. These specialized cores are dedicated to accelerating matrix operations using a non-IEEE754 format (TensorFloat32, or TF32), which only uses 10 bits for the mantissa and is optimized to provide up to 8x speedups over standard FP32 precision. Although this affected the result slightly, it still achieved the $10^(-6)$ required accuracy to validate the benchmark. This matches the performance increase over our CUDA C++ implementation, which is roughly seven times slower than the cuBLAS one.

We also had the opportunity to compare Rust-CUDA and Kokkos (C++) GEMM kernels and obtained comparable performance between implementations.

== Porting partitioning algorithms from a CEA application

#h(1.8em)
The final stage of the internship involves porting parts of a real-world application to the GPU using Rust. This last step aims to push the boundaries of Rust GPGPU programming capabilities and explore the limits of the compiler's help in writing thread-safe kernels. Porting is done using the Rust-CUDA project, targeting NVIDIA GPUs.

=== `coupe`, a concurrent mesh partitioner

#h(1.8em)
The application we chose to port is `coupe` @noauthor_coupe_2023, a modular, multi-threaded library for mesh partitioning written in Rust. It is developed at the CEA/DAM by the joint CEA --- Paris-Saclay University LIHPC laboratory. Coupe implements multiple algorithms aimed at achieving optimal load balancing while minimizing communication costs through the use of geometric methods. 
#linebreak()
Hereafter, we list some of the partitioning algorithms available in the tool, some of which offer optimized variants for cartesian meshes:
- Space-filling curves: Z-curve, Hilbert curve
- Recursive Coordinate Bisection (RCB) and Recursive Inertial Bisection (RIB)
- Multi-jagged
- Karmarkar-Karp
- K-means

=== Recursive Coordinate Bisection (RCB) <rcb>

#h(1.8em)
The algorithm we have chosen to port is the Recursive Coordinate Bisection (RCB) @berger_partitioning_1987 @bramas_novel_2017, one of the simplest geometric algorithms.

#h(1.8em)
Given an N-dimensional set of points, select a vector $n$ of the canonical basis $(e_0, ..., e_(n-1))$. Split the set of points with a hyperplane orthogonal to $n$, such that the two parts of the splits are evenly weighted. Recurse as many times as necessary by reapplying the algorithm to the two parts with another normal vector in each.

#figure(
  image("../../figures/3-contributions/rcb.svg", width: 70%),
  caption: "3-step RCB algorithm visualized on a random set of points"
)<rcb_viz>


@rcb_viz showcases the partitioning of a set of points following the RCB algorithm. We have chosen it because of its straightforward approach and recursive nature, making it ideal for GPU use. 

=== Observations

#h(1.8em)
In practice, trying to port the RCB algorithm was exceedingly difficult. Indeed, most of it relies on a sequence of fundamental algorithms, such as reductions or prefix sums (exclusive scan). Because Rust-CUDA does not have access to library bindings that provide those primitive GPU building blocks, this implied that we had to rewrite everything ourselves. This endeavor proved very time-consuming and challenging to do efficiently by hand. We encountered multiple issues with kernel code generation producing invalid PTX, which meant we had to switch to CUDA C++ implementations instead of Rust-CUDA ones.

In summary, the Rust-CUDA project serves as a robust foundation for crafting "basic" GPU code using Rust. However, it is not currently equipped to handle more complex hardware-accelerated programming tasks, as it lacks many useful abstractions and bindings to libraries that hasten the development of optimized GPU kernels for scientific computing applications.
#linebreak()
We were not able to achieve a working Rust-CUDA port of the RCB algorithm at the time of writing. However, we are investigating a different approach that should simplify the GPU implementation while also better exploiting the architecture of the hardware accelerator. This work will be our main focus for the remainder of the internship.   
