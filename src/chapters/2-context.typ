= Context of the internship

== Hardware accelerators

In this section, we give an overview of hardware accelerators. We introduce the architecture of a GPU, illustrated with current NVIDIA cards, and how they integrate into modern heterogeneous systems. Then, we present the existing programming models used to write code targeting such hardware. Finally, we cover their performance benefits and relevant use cases in HPC.

=== GPU architecture

While CPUs are optimized to compute serial tasks as quickly as possible, GPUs are instead designed to share work between many small processing units that run in parallel. They trade reduced hardware capabilities for handling program logic for much higher core counts that emphasize parallel processing of data. As a result, GPUs prioritize high throughput over low latency, allowing them to outperform CPUs in compute-intensive workloads that can be trivially parallelized, and making them particularly well suited to compute-bound applications.

In this section, we will use the Ampere architecture as an example, as it is described in NVIDIA's whitepaper, and also provide the terminology for equivalent hardware components on AMD GPUs.

#figure(
  image("../../images/2-context/ga100-full.png", width: 100%),
  caption: [Block diagram of the full NVIDIA GA100 GPU implementation],
) <ga100-full>

@ga100-full shows the hierarchy of compute and memory resources available on NVIDIA's data center GA100 GPU/* cite */, designed for HPC and machine learning workloads.

@a100-sm presents the Streaming Multiprocessor (SM) of the GA100 GPU. SMs are the fundamental building block of NVIDIA GPUs and are comparable to what the Compute Unit (CU) is in AMD terminology.
Each SM is a highly parallel processing unit that contains multiple CUDA cores (or shader cores on AMD) and various specialized hardware units. It achieves parallel execution through the Single Instruction, Multiple Thread (SIMT) technique, allowing multiple CUDA cores within the SM to simultaneously execute the same instruction on different data. Threads are scheduled and executed in groups of 32, called "warps" (or "wavefronts" in AMD terminology), thus promoting data parallelism. The SM also manages various types of memory, including fast on-chip registers, instruction and data caches, and shared memory for intra-block communication. Additionally, it provides synchronization mechanisms to coordinate threads within a block.
#linebreak()
Starting from the Volta architecture in 2017/* cite */, NVIDIA SMs introduced an acceleration unit called the Tensor Core, purposefully built for high-performance matrix multiplication and accumulation operations (MMA), which are crucial in AI and machine learning workloads. However, these specialized cores only provide such performance improvements for mixed-precision data types, which reduces their usefulness for most HPC applications that deal with 64-bit precision floating-point. 
On the full implementation of NVIDIA's GA100 GPU, there are 128 SMs and 64 32-bit floating-point CUDA cores per SM, enabling the parallel execution of up to 8192 threads.

#figure(
  image("../../images/2-context/a100-sm.png", width: 40%),
  caption: [Streaming Multiprocessor (SM) of NVIDIA A100 GPU]
) <a100-sm>

NVIDIA GPUs expose multiple levels of memory, each with different capacities, latencies and throughputs. We present them hereafter from fastest to slowest:
#linebreak()
1. *Registers* are the fastest and smallest type of memory available on the GPU, located on the individual CUDA cores (SM units), providing very low latency and extremely fast data access. Registers store local variables and intermediate values during thread execution.
2. *L1 Cache/Shared Memory* is a small, fast, on-chip memory that is shared among threads within the same thread block/* see programming models section*/. This cache level can either be managed automatically by the GPU, or managed manually by the programmer and treated as shared memory between threads. This allows them to communicate and cooperate on shared data. Shared memory is particularly useful when threads need to exchange information or access contiguous data with reduced latency compared to accessing global memory.
3. *L2 Cache* is a larger on-chip memory that serves as a cache for global memory accesses. It is shared among all the SMs in the GPU. L2 cache helps to reduce the latency of global memory accesses by storing recently accessed data closer to the SMs.
4. *Global Memory* is the largest and slowest type of memory available on the GPU as it is located off-chip in the GPU's dedicated Video Random Access Memory (VRAM). Global memory serves as the main memory for the GPU and is used to store data that needs to be accessed by all threads and blocks. However, accessing global memory has higher latency compared to on-chip memories described above. Global memory is generally composed of either Graphic Double Data Rate (GDDR) memory, or of High-Bandwidth Memory (HBM), the latter providing higher throughput in exchange for higher latencies. 
5. *Host Memory (RAM)* refers to the system's main memory, located on the CPU. Data transfers between the CPU and the GPU are necessary for initializing data, transferring results back to the host, or when data does not fit within the GPU's VRAM. Data transfers between host memory and GPU memory often involve much higher latency because of the reduced bandwidth of the bus between the two hardware components (implemented using, e.g., PCIe/*cite*/ buses).

In HPC, the increased integration of GPUs in modern systems requires the use of fast interconnect networks that enable the use of distributed programming models. As most supercomputers use a combination of 2-4 GPUs per CPU (or per socket), there need to be two levels of interconnect fabric:
1. Inter-GPU networks, generally composed of proprietary technologies (e.g., NVLink on NVIDIA, Infinity Fabric on AMD) that ensure the fastest possible data transfers between nearby GPUs.
2. Inter-node networks that allow for fast, OS-free Remote Direct Memory Accesses (RDMA) between faraway GPUs. 

=== Programming models

==== CUDA, HIP, OpenCL

==== SYCL, OpenMP, Kokkos

=== Performance benefits and HPC use cases

Historically, GPUs have primarily been used for graphics-intensive tasks like 3D modeling, rendering, or gaming. However, their highly parallelized design also makes them appealing for HPC workloads which often induce a large number of computations that can be performed concurrently.

== The Rust programming language

=== Language features

=== HPC use cases

=== Interest in GPU programming at the CEA

== Goals

- Establish the state of the art
- Explore the possibilities and limitations compared to vanilla Rust
- Performance comparison
- Proof of concept on CEA applications