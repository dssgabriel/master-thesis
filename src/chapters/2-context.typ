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

= Context of the internship

== Hardware accelerators

#h(1.8em)
In this section, we give an overview of hardware accelerators. We introduce the architecture of a GPU, illustrated with current NVIDIA cards, and how they integrate into modern heterogeneous systems. Then, we present the existing programming models used to write code targeting such hardware. Finally, we cover their performance benefits and relevant use cases in HPC.

=== GPU architecture

#h(1.8em)
While CPUs are optimized to compute serial tasks as quickly as possible, GPUs are instead designed to share work between many small processing units that run in parallel. They trade reduced hardware capabilities in program logic handling, for much higher core counts that emphasize parallel processing of data. As a result, GPUs prioritize high throughput over low latency, allowing them to outperform CPUs in compute-intensive workloads that can be trivially parallelized, and making them particularly well suited to compute-bound applications.

In this section, we will use the Ampere architecture as an example, as it is described in NVIDIA's whitepaper, and also provide the terminology for equivalent hardware components on AMD GPUs.

#figure(
  image("../../figures/2-context/ga100-full.png", width: 100%),
  caption: [Block diagram of the full NVIDIA GA100 GPU implementation],
) <ga100-full>

@ga100-full shows the hierarchy of compute and memory resources available on NVIDIA's data center GA100 GPU/* cite */, designed for HPC and machine learning workloads.

#h(-1.8em)
@a100-sm presents the Streaming Multiprocessor (SM) of the GA100 GPU. SMs are the fundamental building block of NVIDIA GPUs and are comparable to the Compute Unit (CU) in AMD terminology.
Each SM is a highly parallel processing unit containing multiple CUDA cores (or shader cores on AMD) and various specialized hardware units. It achieves parallel execution through the Single Instruction, Multiple Thread (SIMT) technique/*assert that this is true*/, allowing multiple CUDA cores within the SM to simultaneously execute the same instruction on different data. Threads are scheduled and executed in groups of 32, called "warps" (or "wavefronts" in AMD terminology), thus promoting data parallelism. The SM also manages various types of memory, including fast on-chip registers, instruction and data caches, and shared memory for intra-thread block communication. Additionally, it provides synchronization mechanisms to coordinate threads within a block.
#linebreak()
Starting from the Volta architecture in 2017/* cite */, NVIDIA SMs introduced an acceleration unit called the Tensor Core, purposefully built for high-performance matrix multiplication and accumulation operations (MMA), which are crucial in AI and machine learning workloads. However, these specialized cores only provide noticeable performance improvements for mixed-precision data types, reducing their usefulness for most HPC applications that work with double-precision (64-bit) floating-point values. 
On the full implementation of NVIDIA's GA100 GPU, there are 128 SMs and 64 single-precision 
(32-bit) floating-point CUDA cores per SM, enabling the parallel execution of up to 8192 threads.

#figure(
  image("../../figures/2-context/a100-sm.png", width: 40%),
  caption: [Streaming Multiprocessor (SM) of NVIDIA A100 GPU]
) <a100-sm>

NVIDIA GPUs expose multiple levels of memory, each with different capacities, latencies and throughputs. We present them hereafter from fastest to slowest:
#linebreak()
1. *Registers* are the fastest and smallest type of memory available on the GPU, located on the individual CUDA cores (SM units), providing very low latency and extremely fast data access. Registers store local variables and intermediate values during thread execution.
2. *L1 Cache/Shared Memory* is a small, fast, on-chip memory that is shared among threads within the same thread block/*see programming models section*/. This cache level can either be managed automatically by the GPU, or managed manually by the programmer and treated as shared memory between threads. This allows them to communicate and cooperate on shared data. Shared memory is particularly useful when threads need to exchange information or access contiguous data with reduced latency compared to accessing global memory.
3. *L2 Cache* is a larger on-chip memory that serves as a cache for global memory accesses. It is shared among all the SMs in the GPU. L2 cache helps reduce global memory access latency by storing recently accessed data closer to the SMs.
4. *Global Memory* is the largest and slowest type of memory available on the GPU as it is located off-chip in the GPU's dedicated Video Random Access Memory (VRAM). Global memory serves as the main memory for the GPU and is used to store data that needs to be accessed by all threads and blocks. However, accessing global memory has higher latency compared to on-chip memories described above. Global memory is generally composed of either Graphic Double Data Rate (GDDR) memory, or of High-Bandwidth Memory (HBM), the latter providing higher throughput in exchange for higher latencies. 
5. *Host Memory* refers to the system's main memory, i.e., the CPU's RAM. Data transfers between the CPU and the GPU are necessary for initializing data, transferring results back to the host, or when data does not fit within the GPU's global memory. Data transfers between host memory and GPU memory often involve much higher latencies because of the reduced bus bandwidth between the two hardware components (implemented using, e.g., PCIe buses).

@gpu_memory showcases how these kinds of memory are typically organized on an NVIDIA chip.

#figure(
  image("../../figures/2-context/gpu_memory.svg", width: 96%),
  caption: [Generic memory hierarchy of NVIDIA GPUs]
) <gpu_memory>

#h(1.8em)
The increased integration of GPUs in modern HPC systems requires the use of fast interconnect networks that enable the use of distributed programming models. As most supercomputers use a combination of 2-4 GPUs per CPU (or per socket), there need to be two levels of interconnect fabric:
1. Inter-GPU networks, generally comprised of proprietary technologies (e.g., NVLink on NVIDIA, Infinity Fabric on AMD, etc.), ensuring the fastest possible data transfers between nearby GPUs.
2. Inter-node networks that allow for fast, OS-free Remote Direct Memory Accesses (RDMA) between faraway GPUs.

=== Programming models

#h(1.8em)
GPU programming models refer to the different approaches and methodologies used to program and utilize GPUs for general-purpose computation tasks beyond their traditional use cases in graphics rendering. This section introduces some of the programming models used in HPC based on their abstraction level. Firstly, we present low-level models that closely map to the underlying hardware architecture. Secondly, we showcase higher-level programming styles that offer more expressiveness and portability, often at the expense of a high degree of fine-tuned optimization.

We start by introducing common concepts for GPU programming. We define the terms starting from the most high-level view and gradually refine them toward smaller components of GPU programming.

==== Definitions

1. *Kernel:* A kernel is a piece of device code, generally composed of one or multiple functions, that leverages data parallelism (SIMD) and is meant to execute on one or multiple GPUs. A kernel must be launched from host code, although it can be split into multiple smaller kernels that are called from the device.
2. *Grid:* A grid is the highest-level organizational structure of a kernel's execution. It encompasses a collection of blocks, also called work groups and manages their parallel execution. Kernels are launched with parameters that define the configuration of the grid, such as the number of blocks on a given axis. Multiple grids --- i.e., kernels --- can simultaneously exist on a GPU, so to efficiently utilize the available resources.
3. *Block:* A block, also called thread block (CUDA), workgroup (OpenCL), or team/league (OpenMP), is a coarse-grain unit of parallelism in GPU programming. It is the main component of grids and represents a collection of threads that are working together on parts of the data operated on by a kernel. Like grids, the dimensions of blocks can be configured when launching a kernel. Threads in a block can share memory in the L1 cache (see @gpu_memory), which enables better usage of this resource by avoiding expensive, repeated reads to the GPU's global memory.
4. *Warp:* A warp (also called wavefront in AMD terminology) is a fine-grain unit of parallelism in GPU programming, very much related to the hardware implementation of a GPU. However, it also appears at the software level in some programming models. On NVIDIA devices, threads inside a block are scheduled in groups of 32, which programmers can take advantage of in their kernels (e.g., warp-level reductions).
5. *Thread*: A thread is the smallest unit of parallelism in GPU programming. They are grouped in blocks and concurrently perform the operations defined in a kernel. Each thread executes the same instruction as the others but operates on different data (SIMD parallelism).  

@compute_model summarizes these structures as exposed to programmers in a CUDA-style programming model, which we introduce in the next section.

#figure(
  image("../../figures/2-context/compute_model.svg", width: 100%),
  caption: [General compute model for CUDA-style GPU programming]
) <compute_model>

==== Low-level <low_lvl_gpu_prog>

#h(1.8em)
*Computed Unified Device Architecture (CUDA)* /*cite*/ is NVIDIA's proprietary tool for GPU programming. Using a superset of C/C++, it provides a parallel programming platform and several APIs that allow developers to write kernels that will execute on NVIDIA GPUs specifically, locking users into the vendor's ecosystem. However, CUDA is one of the most mature environments for GPU programming, offering a variety of highly optimized computing libraries and tools. Thanks to the vast existing codebase, CUDA is often the default choice for GPU programming in HPC.

*Heterogeneous-Compute Interface for Portability (HIP)* /*cite*/ is the equivalent of CUDA for AMD GPUs. It is part of the RadeonOpenCompute (ROCm) software stack. Contrary to its NVIDIA equivalent, HIP is open-source, making it easier to adopt as it does not lock users into a specific vendor ecosystem. Nevertheless, HIP is a more recent environment than CUDA and as such, it is not as refined. Despite this, HIP provides basic compute libraries (BLAS, DNN, FFT, etc.) optimized for AMD GPUs, and several tools to automatically port CUDA code to HIP's syntax. It is quickly gaining traction as AMD is investing a lot of resources into its GPU programming toolkit in order to catch up with NVIDIA in this space. Its hardware advantage over NVIDIA, from an HPC standpoint, is also enabling AMD to improve adoption in the domain.

*OpenCL* /*cite*/ is developed as an open standard by Khronos Group for low-level, vendor-agnostic GPU programming. Unlike CUDA and HIP, OpenCL also supports FPGA offloading and can even fall back to the CPU if no devices are available. It provides APIs and tools to allow programmers to interact with devices, manage memory, launch parallel executions on the GPU, and write kernels using a superset of C's syntax. The standard defines common mechanisms that make the programming model portable, similar to what @compute_model showcases. Because of its focus on portability, OpenCL implementations can have performance limitations compared to specialized programming models such as CUDA or HIP. Most GPU vendors (NVIDIA, AMD, Intel) supply their own implementation of the OpenCL standard optimized for their hardware. Some open-source implementations of OpenCL and OpenCL compute libraries also exist.

==== High-level

#h(1.8em)
*SYCL* /*cite*/is a recent standard developed by Khronos Group that provides high-level abstractions for heterogeneous programming. Based on recent iterations of the C++ standard (C++17 and above), it aims at replacing OpenCL by simplifying the process of writing kernels and by abstracting the usual low-level compute model (see @compute_model). Kernels written in SYCL look very much like standard CPU code does. However, they can be offloaded to hardware accelerators as desired by the user, in an approach similar to what programmers are used to with OpenCL.

*Open Multi-Processing (OpenMP)* /*cite*/is an API specification for shared-memory parallel programming that also supports offloading to hardware accelerators. It is based on compiler directives for C/C++ and Fortran programming languages. Similarly to SYCL, standard CPU code can be automatically offloaded to the GPU by annotating the relevant section using OpenMP `pragma omp target` clauses. OpenMP is well-known in the field of HPC as it is the primary tool for fine-grain, shared-memory parallelism on CPUs. 

*Kokkos* /*cite*/is a modern C++ performance portability programming ecosystem that provides mechanisms for parallel execution on hardware accelerators. Unlike other programming models, Kokkos implements GPU offloading using a variety of backends (CUDA, HIP, SYCL, OpenMP, etc.). Users write their kernels in standard C++ and can choose their preferred backend for code generation. Kokkos also provides useful memory abstractions, tools, and compute libraries targeted for HPC use cases.   

=== Performance benefits and HPC use cases

Historically, GPUs have primarily been used for graphics-intensive tasks like 3D modeling, rendering, or gaming. However, their highly parallelized design also makes them appealing for HPC workloads which often induce a large number of computations that can be performed concurrently. Applications that are primarily compute-bound can benefit from significant performance improvements when executed on GPUs. Modern HPC systems have already taken advantage of GPUs by tightly incorporating them into their design. Around 98% of the peak performance of modern supercomputers such as Frontier (\#1 machine on the June 2023 TOP500 ranking/*cite*/) comes from GPUs, making it crucial to use them efficiently. 
// Talk about FP precision? AI influence on GPU designs conflicting with HPC interests?

== The Rust programming language

#h(1.8em)
This section introduces the Rust programming language, its notable features, its possible usage in HPC software, and why it is an interesting language for the CEA.

=== Language features

Rust is a compiled, general-purpose, multi-paradigm, imperative, strong, and statically typed language designed for performance, safety and concurrency. Its development started in 2009 at Mozilla, and its first stable release was announced in 2015. As such, it is a rather recent language that aims at becoming the new gold standard for systems programming. Its syntax is based on C and C++ but with a modern twist and heavily influenced by functional languages.

Rust's primary feature that distinguishes it from other compiled languages is its principle of ownership and _borrow-checker_. 
Ownership rules state the following/*cite*/:
1. Each value has an owner.
2. There can only be _one_ owner at a time.
3. When the owner goes out of scope, the value's associated memory is _automatically_ dropped.
Contrarily to C++, there are no implicit deep copies of heap-allocated objects in Rust. Furthermore, variables are constant by default and need to be explicitly declared as mutable, using the `mut` keyword.

#figure(caption: "Rust's ownership in action")[
  ```rs
let s1 = String::from("foo");
let s2 = s1; // `s1` ownership is moved to `s2`, which now owns the value "foo"
println!("{s1}"); // ERROR! value of `s1` has been moved to `s2`
  ```
]<ownership>
In @ownership, declaring `s2` by assigning `s1` to it takes ownership of the value held by `s1` (ownership rule \#2). This invalidates any later use of `s1`, and the Rust compiler can statically catch such mistakes. This guarantees that the compiled code cannot contain use-after-free bugs.

In order to share values between multiple variables, the language also provides references that implement a borrowing mechanism. There are two kinds of references in Rust:
- *Immutable (shared) references* are read-only. Multiple immutable references to the same value can exist simultaneously.
- *Mutable references* allow modifying a value that has been borrowed. However, a mutable reference is unique. There cannot be other references (mutable or shared) to a value that has been mutably borrowed while the reference remains in scope. 
The Rust "borrow-checker" enforces these rules at compile-time. It can also check that any given reference remains valid while in use (i.e., the object it points to is still in scope). This statically guarantees that there are no dangling pointers/references in the code. @borrowing demonstrates these rules annotated with comments summing up the compiler errors, where relevant. 

#figure(caption: "Rust's borrowing in action")[
  ```rs
let s2;
{
    let s1 = String::from("bar");
    s2 = &s1;         // Borrowing the value held by `s1`
    println!("{s1}"); // OK! `s1` has only been borrowed, thus it is still valid
    println!("{s2}"); // OK! `s2` holds a reference to "bar" but does not own it
}
println!("{s2}");     // ERROR! `s2` held a reference that is not valid anymore
                      // because the owner `s1` went out of scope
  ```
  ```rs
let mut s1 = String::from("Hello, "); // `s1` needs to be declared as mutable
                                      // so we can mutably borrow it
{
    let s2 = &mut s1;      // Mutably borrowing the value held by `s1`
    let s3 = &s1;          // ERROR! cannot borrow because `s2` is a mutable
                           // reference in scope, later used to modify `s1`
    s2.push_str("world!"); // OK! Modifying `s1` through `s2`
} // `s2` falls out of scope, and the mutable reference is dropped
let s3 = &s1;              // OK! The are no mutable references to `s1`
println!("{s3}");          // Prints "Hello, world!"

  ```
]<borrowing>

Rust's ownership and borrowing rules eliminate the need for a garbage collector, thus maintaining high performance comparable to other compiled languages such as C, C++ or Fortran. Moreover, they also prevent an entire class of memory safety bugs that plague C/C++ codebases, often causing crashes, memory leaks, or even opening vulnerabilities to cyber-attacks.
#linebreak()
The advantages of these features do not stop there either. By leveraging the rules of ownership and Rust's strict type system, the compiler is able to catch most concurrency-related bugs, such as race conditions or data races.

#figure(caption: "Multi-threaded vector sum in standard C++17")[
  ```cpp
constexpr size_t NELEMENTS = 1'000'000;
constexpr size_t NTHREADS = 8;
constexpr size_t QOT = NELEMENTS / NTHREADS;
constexpr size_t REM = NELEMENTS % NTHREADS;
std::vector<int> vector(NELEMENTS, 1);
std::vector<std::thread> threads(NTHREADS);

int result = 0;
for (size_t t = 0; t < NTHREADS; ++t) {
    size_t const start = t * QOT;
    size_t const end = t == NTHREADS - 1 ? start + QOT + REM : start + QOT;
    threads[t] = std::thread([&]() {
        for (size_t i = start; i < end; ++i) {
            result += vector[i];
        }
    });
}
for (auto& t: threads) {
    t.join();
}

printf("RESULT: %d\n", result);
  ```
]<cpp_thread_safety>

@cpp_thread_safety implements a simple parallel vector sum using C++ standard library threads and a lambda function. The vector contains a million values, all initialized to `1`. Compiling the following code using the following command does not produce any warnings whatsoever:
```
$ g++ -std=c++17 -Wall -Wextra parallel_vector_sum.cpp
```
However, when running the code a few times, we get the following results:
```
$ for i in $(seq 0 5); do ./a.out; done
RESULT: 639810
RESULT: 641278
RESULT: 619719
RESULT: 1235839
RESULT: 590743
```
We obtain different results for each run and never get the expected `1,000,000` result. This is because @cpp_thread_safety contains a race condition that happens when we try to increment the `result` variable.

#figure(
  image("../../figures/2-context/race_cond.svg", width: 60%),
  caption: "Illustration of a race condition"
)<race_cond>

@race_cond shows how two threads, A and B, can cause a race condition while trying to update the value of `result` concurrently. Both threads load the same value and increment it before storing it again. Thread B overwrites the value stored by thread A without taking into account thread A's changes, therefore losing information and producing the wrong sum.

In contrast, Rust's ownership rules allow the compiler to notice such race conditions, making it reject the following equivalent code:

#figure(caption: "Multi-threaded vector sum in standard Rust")[
  ```rs
const NELEMENTS: usize = 1_000_000;
const NTHREADS: usize = 8;
const QOT: usize = NELEMENTS / NTHREADS;
const REM: usize = NELEMENTS % NTHREADS;
let vector = vec![1; NELEMENTS];
let mut threads = Vec::with_capacity(NTHREADS);

let mut result = 0;
for t in 0..NTHREADS {
    let start = t * QOT;
    let end = if t == NTHREADS - 1 {
        start + QOT + REM
    } else {
        start + QOT
    };
    threads.push(std::thread::spawn(|| {
        for i in start..end {
            result += vector[i]; // thread `t` mutably borrows `result`
        }
    }));
}
for t in threads {
    t.join().unwrap();
}

println!("RESULT: {result}");
  ```
]<rs_thread_safety>

Indeed in @rs_thread_safety, although Rust automatically infers that it must mutably borrow `result` in the thread's lambda (called "closures" in Rust), it cannot guarantee that the thread will finish executing before `result` goes out of scope. Furthermore, when a thread `t` mutably borrows `result`, it prevents the other threads from borrowing it, resulting in a compiler error. The full error message is available in @error_race_cond in the @appendix.
#linebreak()
In some cases, Rust can even propose the relevant changes to make the code valid. E.g., @rs_thread_safety can be fixed by either:
- Making the `result` variable atomic, in order to guarantee shared-memory communication of the updated value between threads;
- Wrapping the `result` variable with a lock (e.g., a mutex) to ensure that increments of the value are protected atomically.

Rust's ownership and borrowing rules make it a great fit for parallel programming, as the compiler can assert that the code is semantically correct and will produce the expected behavior. Being a compiled language, it is able to match, and even sometimes surpass the performance of its direct competitors: C and C++.

#h(-1.8em)
Hereafter we exhaustively list other useful features that the language includes but that are not worth exploring in detail in the available space of this report:
- Type inference;
- Smart pointers and RAII mechanisms for automatic resource allocation and deallocation;
- Powerful enum types that can both encode meaning and hold values, paired with pattern-matching expressions to concisely handle variants;
- Optional datatypes that enable compact error handling for both recoverable and unrecoverable errors;
- A generic typing system, and _traits_ that provide ways to express shared behaviors between objects;
- A robust documenting, testing, and benchmarking framework integrated into the language;
- A broad standard library that provides an extensive set of containers, algorithms, OS and I/O functionalities, asynchronous and networking primitives, concurrent data structures and message passing features, etc.

Rust also comes with a vast set of tools to aid software development:
- A toolchain manager that handles various hardware targets, `rustup`;
- A package manager and build system, `cargo`;
- A registry for sharing open-source libraries, `crates.io`;
- A comprehensive language and standard library documentation, `docs.rs`;
- First-class programming tools for improved development efficiency: a language server, `rust-analyzer`, a linter `clippy`, a code formatter `rustfmt`, etc.

=== HPC use cases

The accent that Rust puts on performance, safety and concurrency makes the language a fitting candidate for becoming a first-tier choice for HPC applications. Its focus on thread safety, in particular, empowers programmers to write fast, robust, and safe code that will be easily maintainable and improvable over its lifetime. Rust avoids many of the pitfalls of C++, especially in terms of language complexity, and its modern features and syntax make it a lot easier to work with than Fortran. Its adoption into many of the top companies that work in fields related to HPC (Amazon, Google, Microsoft Azure, etc.), and its acceptance as the second language of the Linux kernel helped it gain a lot of traction in low-level, high-performance programming domains. Not only is it well suited to writing scientific software that relies on efficient parallelism, Rust is also a formidable language for writing HPC tools, such as profilers, debuggers, or even low-level libraries that power abstractions in higher-level languages (e.g., Python, Julia, etc.).

== Goals

#h(1.8em)
The aim of this internship is to establish an exhaustive state of the art for the capabilities of Rust in GPU programming. The goal is to explore what the language is currently able to natively support, and what are the existing frameworks or libraries for writing GPGPU software.

There are several crates, e.g., `rayon`, that enable trivial parallelization of code sections for CPU use cases, similar to OpenMP's ease of use in C, C++ and Fortran. This library provides parallel implementations of Rust's standard iterators that fully leverage the language thread-safety features. Code is guaranteed to be correct at compile time, and unlocks the processor's maximum multi-core performance. `rayon` also implements automatic work-stealing techniques that keep the CPU busy, even when the application's load balancing is not optimal. We want to investigate if something similar exists for GPU computing, and, if not, to determine what the limitations would be if we tried to.

In a secondary stage, we want to assess Rust's ability to keep up with the performance of C and C++ GPGPU programming. This comparison would be primarily based on common compute kernels and should aim at evaluating what are the best options for GPU code generation in Rust.

Finally, we would like to research the limits of Rust for GPU computing by porting parts of real-world CEA applications. This work involves evaluating both the effort necessary for such ports, as well as the performance improvements that we can expect for industrial-grade software.
