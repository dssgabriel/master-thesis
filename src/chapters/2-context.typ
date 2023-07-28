= Context of the internship

== Hardware accelerators

In this section, we give an overview of hardware accelerators. We introduce the architecture of a GPU, illustrated with current NVIDIA cards, and how they integrate into modern heterogeneous systems. Then, we present the existing programming models used to write code targeting such hardware. Finally, we cover their performance benefits and relevant use cases in HPC.

=== GPU architecture

While CPUs are optimized to compute serial tasks as quickly as possible, GPUs are designed to share the work between many small cores that run in parallel. GPUs, therefore, trade low latency for high throughput, often outperforming CPUs in compute-intensive workloads where the operations can be done concurrently. Hence, they feature very high core counts and reduced hardware capabilities for program logic. GPUs are particularly well suited to compute-bound applications.

// <Insert GPU architecture figure>

==== Hardware and memory hierarchy
Most GPU architectures have several levels of parallelism. Specifically, on NVIDIA cards, we have four levels:
1. the grid/SM (needs clarification)
2. the thread block
3. the warp
4. the thread

Definition 1: Streaming Multi-processor (SM)

Definition 2: Thread Block

Definition 3: Warp

Tensor cores

Memory:
- Global memory (GDDR, HBM)
- Local memory (L2, L1)
- Private memory (L1, L0, registers)

==== Interconnect networks and heterogeneous systems integration
Standards, PCIe, Infiniband, NVLink, etc.
HPC systems use 3/4 GPUs per CPU/socket, generally with 2 sockets per node.

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
