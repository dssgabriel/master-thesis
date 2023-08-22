= Context of the internship

== Hardware accelerators

#h(1.8em)
In this section, we give an overview of hardware accelerators. We introduce the architecture of a GPU, illustrated with current NVIDIA cards, and how they integrate into modern heterogeneous systems. Then, we present the existing programming models used to write code targeting such hardware. Finally, we cover their performance benefits and relevant use cases in HPC.

=== GPU architecture

#h(1.8em)
While CPUs are optimized to compute serial tasks as quickly as possible, GPUs are instead designed to share work between many small processing units that run in parallel. They trade reduced hardware capabilities in program logic handling, for much higher core counts that emphasize parallel processing of data. As a result, GPUs prioritize high throughput over low latency, allowing them to outperform CPUs in compute-intensive workloads that can be trivially parallelized, and making them particularly well suited to compute-bound applications.

In this section, we will use the Ampere architecture as an example, as it is described in NVIDIA's whitepaper, and also provide the terminology for equivalent hardware components on AMD GPUs.

#figure(
  image("../../images/2-context/ga100-full.png", width: 100%),
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
  image("../../images/2-context/a100-sm.png", width: 40%),
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
  image("../../images/2-context/gpu_memory.svg", width: 96%),
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
2. *Grid:* A grid is the highest-level organizational structure of a kernel's execution. It encompasses a collection of blocks, also called work groups and manage
s their parallel execution. Kernels are launched with parameters that define the configuration of the grid, such as the number of blocks on a given axis. Multiple grids --- i.e., kernels --- can simultaneously exist on a GPU, so to efficiently utilize the available resources.
3. *Block:* A block, also called thread block (CUDA), workgroup (OpenCL), or team/league (OpenMP), is a coarse-grain unit of parallelism in GPU programming. It is the main component of grids and represents a collection of threads that are working together on parts of the data operated on by a kernel. Like grids, the dimensions of blocks can be configured when launching a kernel. Threads in a block can share memory in the L1 cache (see @gpu_memory), which enables better usage of this resource by avoiding expensive, repeated reads to the GPU's global memory.
4. *Warp:* A warp (also called wavefront in AMD terminology) is a fine-grain unit of parallelism in GPU programming, very much related to the hardware implementation of a GPU. However, it also appears at the software level in some programming models. On NVIDIA devices, threads inside a block are scheduled in groups of 32, which programmers can take advantage of in their kernels (e.g., warp-level reductions).
5. *Thread*: A thread is the smallest unit of parallelism in GPU programming. They are grouped in blocks and concurrently perform the operations defined in a kernel. Each thread executes the same instruction as the others but operates on different data (SIMD parallelism).  

@compute_model summarizes these structures as exposed to programmers in a CUDA-style programming model, which we introduce in the next section.

#figure(
  image("../../images/2-context/compute_model.svg", width: 100%),
  caption: [General compute model for CUDA-style GPU programming]
) <compute_model>

==== Low-level

#h(1.8em)
*Computed Unified Device Architecture (CUDA)* /*cite*/ is NVIDIA's proprietary tool for GPU programming. Using a superset of C/C++, it provides a parallel programming platform and several APIs that allow developers to write kernels that will execute on NVIDIA GPUs specifically, locking users into the vendor's ecosystem. However, CUDA is one of the most mature environments for GPU programming, offering a variety of highly optimized computing libraries and tools. Thanks to the vast existing codebase, CUDA often is the default choice for GPU programming in HPC.

*Heterogeneous-Compute Interface for Portability (HIP)* /*cite*/ is the equivalent of CUDA for AMD GPUs. It is part of the RadeonOpenCompute (ROCm) software stack. Contrary to its NVIDIA equivalent, HIP is open-source, making it easier to adopt as it does not lock users into a specific vendor ecosystem. Nevertheless, HIP is a more recent environment than CUDA and as such, it is not as refined. Despite this, HIP provides basic compute libraries (BLAS, DNN, FFT, etc.) optimized for AMD GPUs, and several tools to automatically port CUDA code to HIP's syntax. It is quickly gaining traction as AMD is investing a lot of resources into its GPU programming toolkit in order to catch up with NVIDIA in this space. Its hardware advantage over NVIDIA, from an HPC standpoint, is also enabling AMD to improve adoption in the domain.

*OpenCL* /*cite*/ is developed as an open standard by Khronos Group for low-level, vendor-agnostic GPU programming. Unlike CUDA and HIP, OpenCL also supports FPGA offloading and can even fall back to the CPU if no devices are available. It provides APIs and tools to allow programmers to interact with devices, manage memory, launch parallel executions on the GPU, and write kernels using a superset of C's syntax. The standard defines common mechanisms that make the programming model portable, similar to what @compute_model showcases. Because of its focus on portability, OpenCL implementations can have performance limitations compared to specialized programming models such as CUDA or HIP. Most GPU vendors (NVIDIA, AMD, Intel) supply their own implementation of the OpenCL standard optimized for their hardware. Some open-source implementations of OpenCL and OpenCL compute libraries also exist.

==== High-level

#h(1.8em)
*SYCL* /*cite*/is a recent standard developed by Khronos Group that provides high-level abstractions for heterogeneous programming. Based on recent iterations of the C++ standard (C++17 and above), it aims at replacing OpenCL by simplifying the process of writing kernels and by abstracting the usual low-level compute model (see @compute_model). Kernels written in SYCL look very much like standard CPU code does. However, they can be offloaded to hardware accelerators as desired by the user, in an approach similar to what programmers are used to with OpenCL.

*Open Multi-Processing (OpenMP)* /*cite*/is an API specification for shared-memory parallel programming that also supports offloading to hardware accelerators. It is based on compiler directives for C/C++ and Fortran programming languages. Similarly to SYCL, standard CPU code can be automatically offloaded to the GPU by annotating the relevant section using OpenMP `pragma omp target` clauses. OpenMP is well-known in the field of HPC as it is the primary tool for fine-grain, shared-memory parallelism on CPUs. 

*Kokkos* /*cite*/is a modern C++ performance portability programming ecosystem that provides mechanisms for parallel execution on hardware accelerators. Unlike other programming models, Kokkos implements GPU offloading using a variety of backends (CUDA, HIP, SYCL, OpenMP, etc.). Users write their kernels in standard C++ and can choose their prefered backend for code generation. Kokkos also provides useful memory abstractions, tools and compute libraries target for HPC use cases.   

=== Performance benefits and HPC use cases

Historically, GPUs have primarily been used for graphics-intensive tasks like 3D modeling, rendering, or gaming. However, their highly parallelized design also makes them appealing for HPC workloads which often induce a large number of computations that can be performed concurrently. Applications that are primarily compute-bound can benefit from significant performance improvements when executed on GPUs. Modern HPC systems have already taken advantage of GPUs by tightly incorporating them into their design. Around 98% of the peak performance of supercomputers such as Frontier (\#1 machine on the June 2023 TOP500 ranking/*cite*/) comes from GPUs, making it crucial to use them efficiently. 
// Talk about FP precision? AI influence on GPU designs conflicting with HPC interests?

== The Rust programming language

#h(1.8em)
This section introduces the Rust programming language, its notable features, its possible usage in HPC software, and why it is an interesting language for the CEA.

Rust is a general-purpose, multi-paradigm, imperative, strong and statically typed language designed for performance, safety and concurrency. Its development started in 2009 at Mozilla, and its first stable release was announced in 2015. As such, it is a rather recent language that aims at becoming the new gold standard for systems programming. Its syntax is based on C and C++ but with a modern twist and heavily influenced by functional languages.

#show raw.where(block: true): block.with(
  fill: luma(240),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
)

#figure(caption: "Simple vector sum in Rust")[
```rs
fn main() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 5.0];
    let sum: f64 = x.iter().sum();
    println!("x  = {x:?}");
    println!("Σx = {sum}");
}
```
```
x  = [0.0, 1.0, 2.0, 3.0, 5.0]
Σx = 11
```
]

=== Language features

```rs
const NPARTICLES: usize = 100;
let particles_positions = vec![Particle::default(); NPARTICLES];
let mut particles_forces = vec![Particle::default(); NPARTICLES];

particles_forces
    .par_iter_mut()
    .enumerate()
    .map(|(i, particles_forces_i)| {
        for j in (i + 1)..NPARTICLES {
            let distance = R_STAR
                / distance(particles_positions[i], particles_positions[j]);

            // Compute force exerted on particle `i`
            let force = distance
                * (particles_positions[i] - particles_positions[j]);

            // Update forces
            *particles_forces_i += force;
            particles_forces[j] -= force;
        }
    });
```

#pagebreak()
```
error[E0596]: cannot borrow `particles_forces` as mutable, as it is a captured variable in a `Fn` closure
  --> src/main.rs:29:17
   |
29 |                 particles_forces[j] -= force;
   |                 ^^^^^^^^^^^^^^^^ cannot borrow as mutable

error[E0499]: cannot borrow `particles_forces` as mutable more than once at a time
  --> src/main.rs:20:14
   |
17 | /     particles_forces
18 | |         .par_iter_mut()
   | |_______________________- first mutable borrow occurs here
19 |           .enumerate()
20 |           .map(|(i, particles_forces_i)| {
   |            --- ^^^^^^^^^^^^^^^^^^^^^^^^^ second mutable borrow occurs here
   |            |
   |            first borrow later used by call
...
29 |                   particles_forces[j] -= force;
   |                   ------------------- second borrow occurs due to use of
   |                                       `particles_forces` in closure
```

=== HPC use cases

=== Interest in GPU programming at the CEA

== Goals

- Establish the state of the art
- Explore the possibilities and limitations compared to vanilla Rust
- Performance comparison
- Proof of concept on CEA applications
