= Introduction

#h(1.8em)
In the last ten years, HPC has experienced a dramatic shift in computer architecture, slowly moving away from general-purpose central processing units (CPU) and instead turning towards heterogeneous systems with specialized hardware designed to accelerate computations. The use of graphical processing units (GPU), field programmable gate arrays (FPGA), or even application-specific integrated circuits (ASIC) has increased significantly in modern supercomputers, often outnumbering CPUs by a factor of four in the systems that have most recently entered the TOP500 ranking. This change leads to a growing need for efficient software that exploits the computational performance unlocked by such accelerators. Even more recently, the surging of artificial intelligence (AI) has pushed performance requirements even further with extensive reliance on GPU architectures, which are especially well-suited for these workloads. This leads to a rapid convergence between HPC and AI, in which both fields depend on similar hardware but accommodate different computational demands.

To meet these new criteria and take advantage of the performance improvements offered by GPUs, the software has to change, which involves rewriting significant portions of existing applications. This endeavor is not trivial, and fully exploiting these accelerators requires comprehensive knowledge of GPU architecture. Moreover, rewrites often induce complex communications between CPU and GPU to transfer the data between their respective memory space. These come at a high cost that is difficult to offset, as memory latency and bandwidth have improved very slowly compared with the hardware computing performance. In the pursuit of efficiency, programming languages offer modern tools to work with accelerators, either by offering low-level control of the device (e.g., CUDA C++, OpenCL C, etc.) or by providing higher-level concepts that abstract over architectural details, sometimes sacrificing performance in favor of better code portability (e.g., SYCL, Kokkos, OpenMP, etc.).

The Rust programming language is a newcomer in the field of high-performance compiled languages, with its first stable release in 2015. It aims to solve most of the memory and type safety issues that exist in C and C++ while maintaining equivalent performance. It also puts a significant accent on correctness in concurrency contexts by eliminating an entire class of data race bugs thanks to its borrow checker. Rust thus provides robust safety guarantees without performance penalties, packed in a modern syntax with many functional features that align well with the current trends in software engineering.

This internship aims to evaluate the viability of Rust as a GPGPU programming language in the context of scientific computing and HPC. In particular, the goal is to determine if we can leverage some of the language's properties to guarantee the robustness, memory and thread safety of GPU codes developed at the CEA.
