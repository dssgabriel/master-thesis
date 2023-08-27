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

There is an open tracking issue for PTX code generation problems/*cite*/ but there has not been any contribution to it since March 2022. Rust's efforts for GPU programming native support seems to currently be at a stop.

=== Shading languages and external libraries

Shading languages seem to be the most popular approach for programming GPUs using the Rust language. There are multiple actively maintained crates that offer support for writing GPU code in Rust. Hereafter, we list a few of the most relevant and active ones:
- `rust-gpu` /*cite*/
- `wgpu-rs` /*cite*/
- `vulkano-rs` /*cite*/

=== OpenCL

=== CUDA

== Open-source work on the Rust-CUDA project

== Hardware-Accelerated Rust Profiling

=== Benchmark methodology

=== Comparison of GPU code generation methods

=== Results analysis

== Porting partitioning algorithms from a CEA application

=== `coupe`, a concurrent mesh partitioner

=== Recursive Coordinate Bisection (RCB)

=== Observations