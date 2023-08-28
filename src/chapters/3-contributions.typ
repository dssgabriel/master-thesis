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

=== Native language support

The Rust programming language officially supports NVIDIA's `nvptx64` architecture as a tier 2/*cite*/ compiler target/*cite*/. This includes support for the following :
- Writing kernels directly in Rust
- Intrinsics for retrieving a thread's unique identifier
- Synchronization primitives for block-level scheduling

However, this initial support is very limited compared to writing standard Rust. Indeed, kernels cannot depend on Rust's standard library. Kernels must be declared as `unsafe` functions, which reduces the ability of the compiler to assert the correctness of the GPU code. Moreover, one of Rust's most useful abstractions, slices, are not usable inside functions compiled for the `nvptx64` target. This forces the use of raw pointers in order to interact with memory buffers. As pointer arithmetic is forbidden in Rust, it is necessary to use the core `add` method to correctly offset a pointer's address before de-referencing it. These shortcomings result in highly verbose kernel code, which is hard to both read and write.

#figure(caption: "Minimal example for writing a native Rust DAXPY GPU kernel")[
  ```rust
#![no_std]
#![no_main]
#![deny(warnings)]
#![feature(abi_ptx, core_intrinsics)]

use core::arch::nvptx;

#[no_mangle]
pub unsafe extern "ptx-kernel" fn daxpy_kernel(
    x: *const f64,
    alpha: f64,
    y: *mut f64
) {
    let idx = (nvptx::_block_idx_x() * nvptx::_block_dim_x()
        + nvptx::_thread_idx_x()) as usize;
    let item = &mut y.add(idx);
    *item = alpha * *&x.add(idx);
}

#[panic_handler]
unsafe fn breakpoint_panic_handler(_: &::core::panic::PanicInfo) -> ! {
    core::intrinsics::breakpoint();
    core::hint::unreachable_unchecked();
}
  ```
]<rustc_nvptx>

As @rustc_nvptx demonstrates, a kernel as simple as DAXPY is unnecessarily verbose to write. This makes GPU code exceedingly difficult to work with in native Rust due to the high amount of complexity implied by working with no language abstractions.
#linebreak()
Furthermore, the current Rust compiler (rustc v1.72.0) is unable to produce a valid executable of the above code snippet. The CUDA runtime will throw an error saying that the provided PTX assembly (NVIDIA's proprietary high-level assembly language) is invalid when trying to load it.

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

As mentioned in @low_lvl_gpu_prog, OpenCL is a low-level GPU programming model. There are two Rust crates that provide bindings to the OpenCL 3 API: `cogciprocate/ocl`/*cite*/ and `kenba/opencl3`.
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

=== CUDA<cuda>

CUDA is a low-level proprietary GPU programming language specifically designed for NVIDIA hardware accelerators.

== Open-source work on the Rust-CUDA project

In order to support the latest major release of the NVIDIA CUDA Toolkit (version 12), we had to fix the breaking changes introduced between this version and the last stable version with which Rust-CUDA was compatible, CUDA 11.8. As mentioned in @cuda, the Rust-CUDA project uses a custom compiler backend, `rustc-codegen-nvvm`, which depends on NVIDIA's NVVM IR. As part of the CUDA 12 Toolkit update, a new version of the NVVM IR specification was released.
#linebreak()
NVVM 2.0 introduced the following breaking changes:
- Removed address space conversion intrinsics.
- Stricter error checking on the supported data layouts.
- Older style loop unroll pragma metadata on loop back edges is no longer supported.
- Shared variable initialization with non-undefined values is no longer supported.

== Hardware-Accelerated Rust Profiling

=== Benchmark methodology

=== Comparison of GPU code generation methods

=== Results analysis

== Porting partitioning algorithms from a CEA application

=== `coupe`, a concurrent mesh partitioner

=== Recursive Coordinate Bisection (RCB)

=== Observations