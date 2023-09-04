= Conclusion

#h(1.8em)
This internship aimed to explore Rust's viability for GPGPU programming in scientific computing use cases. On the one hand, HPC is shifting towards an ever-growing reliance on hardware accelerators, especially GPUs, and it is crucial to write robust code that efficiently exploits these new heterogeneous architectures. On the other hand, the Rust programming language focuses on performance, safety and concurrency, three aspects that align well with HPC's needs. Rust's type, memory, and thread safety features make it a particularly appealing language for this task.

Without any prior comprehensive work on Rust's potential for GPU programming, we set out to establish an exhaustive state of the art for the available options in this domain. We investigated various possibilities, first starting with the language's native support. Then, we explored compute shaders libraries that brought hardware accelerator programming capabilities to Rust. We concluded that the most relevant strategies for developing GPGPU kernels involved using bindings to the OpenCL API and the Rust-CUDA project, designed to bring first-class support for NVIDIA's CUDA framework within the Rust ecosystem.
#linebreak()
During this internship, we also contributed to the Rust-CUDA project by updating the code generation pipeline to support the newly released CUDA 12 Toolkit. We also added support for the most recent NVIDIA architectures to optimize the generated PTX targeting them.

The next step of this internship was to develop a scientific approach to evaluate the performance of Rust-based GPU programming. To this end, we developed an open-source tool, HARP, which automates the performance benchmarking of basic kernels often encountered in scientific applications. HARP was carefully designed to provide accurate and reliable results and remain portable across a wide variety of systems, from small laptops to bleeding-edge supercomputers.

Finally, we pushed the boundaries of Rust-based GPU programming by attempting to port parts of an industrial-grade scientific library for mesh partitioning. This proved to be remarkably challenging as we reached the limits of the Rust-CUDA project. Numerous constraints constitute critical opportunities for improving Rust's support in GPU programming.

The main challenge for Rust's adoption remains the low amount of contributions in projects that focus on HPC and scientific computing. This is reinforced by the absence of libraries that provide the basic blocks for writing more complex and well-optimized algorithms on hardware accelerators (e.g., NVIDIA CUB and NVIDIA Thrust in the CUDA ecosystem).

However, while the prospect of developing an entire application leveraging GPUs in Rust may currently seem challenging, the language exhibits several promising features that could turn into serious assets in the future.  Rust already largely outclasses C and C++ for orchestrating the environment surrounding GPU execution. This includes the management of devices, streams, memory allocations and transfers, kernel launches, and more. Rust's distinctive memory management approach significantly alleviates the challenges posed by error handling and resource deallocation in C/C++. These often result in a plethora of elusive memory bugs, which can be arduous to trace and rectify. The OpenCL API bindings and the `cust` crate within the Rust-CUDA project already constitute a significantly better alternative than their default C/C++ counterparts.

Furthermore, as we demonstrated through several benchmarks presented in this report, Rust is capable of rivaling and even exceeding the performance of C++. Although kernels written using Rust-CUDA still lack some of the usual abstractions traditionally offered by Rust, they are more compact than their CUDA C++ equivalents. Given that Rust-CUDA is still in its early stages of development, it is easy to envision a future where Rust is the most accessible, efficient, and robust choice for GPU programming.

== Perspectives and future work

#h(1.8em)
The final weeks of this internship will focus on finishing to implement the GPU version of the RCB algorithm discussed in @rcb. We will also get the chance to conduct more benchmarks on an NVIDIA A100 GPU, which we hope to present during the thesis defense.

Beyond the scope of this internship, Rust has several challenges to address if it aims to become a proper first-class language for GPU programming. Its primary goal should be to improve the native language support, at least make it a usable alternative to Rust-CUDA. The Rust project leadership could also consider integrating the Rust-CUDA project into the upstream rustc compiler so as to benefit from the advanced work that has already been done and improve upon it. Likewise, the focus should be on enhancing the existing abstractions, particularly for manipulating slices, as well as fixing some of the overly verbose syntax required to write kernels. 

Moreover, the serious lack of compute-focused libraries for GPU programming should be tackled as soon as possible to make Rust a more viable language for writing kernels.
#linebreak()
There are two possible approaches we can already propose:
1. Write bindings for existing C/C++ libraries that already fit this purpose, e.g., NVIDIA CUB and NVIDIA Thrust.
2. Implement an idiomatic wrapper over Rust-based kernels that provide the same functionality as Rust's native standard library, notably for handling iterators. This should allow for writing kernels that look and feel very much like usual CPU code while also benefiting from highly optimized GPU primitives that are abstracted away thanks to "syntax sugar."

#h(1.8em)
Another point that could be explored is Rust's support and performance on AMD hardware accelerators. This internship has been predominantly focused on NVIDIA GPUs, as it was the only hardware we had available. Future work could investigate the performance of OpenCL Rust on AMD, especially compared to ROCm/HIP C++. Likewise, we should examine the viability of a Rust-ROCm project, similar to what we have with Rust-CUDA, and even consider a direct integration into the Rust compiler.