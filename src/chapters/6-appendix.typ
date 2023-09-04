#show raw.where(block: true): it => {
    set text(font: "IBM Plex Mono")
    set align(left)
    set block(fill: luma(240), inset: 10pt, radius: 4pt, width: 100%)
    it
}

= Appendix <appendix>

== Glossary

Items in the glossary are ordered alphabetically.

#set par(first-line-indent: 0em)
#set terms(separator: ": ", tight: true, spacing: auto)
/ ABI: Application Binary Interface
/ AI: Artificial Intelligence
/ API: Application Programming Interface
/ ASIC: Application-Specific Integrated Circuit
/ AXPY: $alpha x + y$, vector addition
/ BLAS: Basic Linear Algebra Software
/ CU: Compute Unit
/ CUDA: Compute Unified Device Architecture
/ (G)DDR: (Graphics) Double Data Rate (memory)
/ DNN: Deep Neural Network
/ FMA: Fused Multiply-Add
/ FP: Floating Point
/ FPGA: Field Programmable Gate Array
/ FFT: Fast Fourier Transform
/ GEMM: General Matrix Multiplication
/ (GP)GPU: (General Purpose) Graphics Programming Unit
/ HBM: High-Bandwidth Memory
/ HIP: Heterogeneous-Compute Interface for Portability
/ HPC: High Performance Computing
/ I/O: Input/Output
/ IR: Intermediate Representation
/ ISA: Instruction Set Architecture
/ JIT: Just-In-Time (compilation)
/ MMA: Matrix Multiply-Accumulate
/ OpenCL: Open Computing Language
/ OpenMP: Open Multi-Processing
/ OS: Operating System
/ PCIe: Peripheral Component Interconnect Express
/ PTX: Parallel-Thread eXecution
/ RAII: Resource Acquisition Is Initialization
/ (V)RAM: (Video) Random Access Memory
/ RCB: Recursive Coordinate Bisection
/ RDMA: Remote Direct Memory Access
/ RIB: Recursive Inertial Bisection
/ ROCm: Radeon Open Compute
/ SASS: Streaming ASSembler
/ SIMD: Single Instruction, Multiple Data
/ SIMT: Single Instruction, Multiple Thread
/ SM: Streaming Multiprocessor
/ TF: Tensor Float

#pagebreak()
== Listings 

#figure(caption: "Rust's compiler error message for a race condition bug in")[
  ```
error[E0373]: closure may outlive the current function, but it borrows `result`, which is owned by
the current function
  --> src/thread_safety.rs:18:41
   |
18 |         threads.push(std::thread::spawn(|| {
   |                                         ^^ may outlive borrowed value `result`
19 |             for i in start..end {
20 |                 result += array[i];
   |                 ------ `result` is borrowed here
   |
note: function requires argument type to outlive `'static`
  --> src/thread_safety.rs:18:22
   |
18 |           threads.push(std::thread::spawn(|| {
   |  ______________________^
19 | |             for i in start..end {
20 | |                 result += array[i];
21 | |             }
22 | |         }));
   | |__________^
help: to force the closure to take ownership of `result` (and any other referenced variables), use
the `move` keyword
   |
18 |         threads.push(std::thread::spawn(move || {
   |                                         ++++

error[E0499]: cannot borrow `result` as mutable more than once at a time
  --> src/thread_safety.rs:18:41
   |
18 |           threads.push(std::thread::spawn(|| {
   |                        -                  ^^ `result` was mutably borrowed here in the previous iteration of the loop
   |  ______________________|
   | |
19 | |             for i in start..end {
20 | |                 result += array[i];
   | |                 ------ borrows occur due to use of `result` in closure
21 | |             }
22 | |         }));
   | |__________- argument requires that `result` is borrowed for `'static`
  ```
]<error_race_cond>

#figure(caption: "Minimal OpenCL C code that builds and run an OpenCL DAXPY kernel on a GPU", caption-pos: top)[
  ```c
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef MAC
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#define PROGRAM_FILE "add_numbers.cl"
#define KERNEL_FUNC "add_numbers"
#define ARRAY_SIZE 64

cl_device_id create_device() {
    cl_platform_id platform;
    cl_device_id dev;
    int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0) {
        perror("Couldn't identify a platform");
        exit(1);
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err == CL_DEVICE_NOT_FOUND) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    if (err < 0) {
        perror("Couldn't access any devices");
        exit(1);
    }

    return dev;
}

cl_program build_program(cl_context ctx, cl_device_id dev, char const* filename) {
    cl_program program;
    FILE* program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    int err;

    program_handle = fopen(filename, "r");
    if (program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);
```
]<ocl_c>

#pagebreak()

```c
    program =
        clCreateProgramWithSource(ctx, 1, (char const**)(&program_buffer), &program_size, &err);
    if (err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);


    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0) {
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*)(malloc(log_size + 1));
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}

int main() {
    cl_device_id device;
    cl_context context;
    cl_program program;
    cl_kernel kernel;
    cl_command_queue queue;
    cl_int i, j, err;
    size_t local_size, global_size;

    float data[ARRAY_SIZE];
    float sum[2], total, actual_sum;
    cl_mem input_buffer, sum_buffer;
    cl_int num_groups;

    for (i = 0; i < ARRAY_SIZE; i++) {
        data[i] = 1.0f * i;
    }

    device = create_device();
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err < 0) {
        perror("Couldn't create a context");
        exit(1);
    }

    program = build_program(context, device, PROGRAM_FILE);

    global_size = 8; // WHY ONLY 8?
    local_size = 4;
    num_groups = global_size / local_size;
    input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  ARRAY_SIZE * sizeof(float), data, &err);
    if (err < 0) {
        perror("Couldn't create a buffer");
        exit(1);
    };

    sum_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                num_groups * sizeof(float), sum, &err);
    if (err < 0) {
        perror("Couldn't create a buffer");
        exit(1);
    };

    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err < 0) {
        perror("Couldn't create a command queue");
        exit(1);
    };

    kernel = clCreateKernel(program, KERNEL_FUNC, &err);
    if (err < 0) {
        perror("Couldn't create a kernel");
        exit(1);
    };

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(kernel, 1, local_size * sizeof(float), NULL);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &sum_buffer);
    if (err < 0) {
        perror("Couldn't create a kernel argument");
        exit(1);
    }

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    if (err < 0) {
        perror("Couldn't enqueue the kernel");
        exit(1);
    }

    err = clEnqueueReadBuffer(queue, sum_buffer, CL_TRUE, 0, sizeof(sum), sum, 0, NULL, NULL);
    if (err < 0) {
        perror("Couldn't read the buffer");
        exit(1);
    }

    clReleaseKernel(kernel);
    clReleaseMemObject(sum_buffer);
    clReleaseMemObject(input_buffer);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 0;
}
  ```

#figure(caption: "Minimal OpenCL Rust code for building and launching an OpenCL DAXPY kernel on a GPU, using the `ocl` crate", caption-pos: top)[
  ```rs
extern crate ocl;
use ocl::ProQue;

const ARRAY_SIZE: usize = 64;
static SRC: &str = r#"
__kernel void fma(__global float const* input, float scalar, __global float* output) {
    int idx = get_global_id(0);
    output[idx] += scalar * output[idx];
}
"#;

fn main() -> ocl::Result<()> {
    let pro_que = ProQue::builder().src(SRC).dims(ARRAY_SIZE).build()?;
    let input = pro_que.create_buffer::<f32>()?;
    let d_output = pro_que.create_buffer::<f32>()?;

    let kernel = pro_que
        .kernel_builder("fma")
        .arg(&input)
        .arg(10.0f32)
        .arg(&d_output)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    let mut h_output = vec![0.0f32; d_output.len()];
    d_output.read(&mut h_output).enq()?;

    Ok(())
}
  ```
]<ocl_rs>

#pagebreak()
== Figures <anx_figs>

#figure(
  image("../../figures/6-appendix/saxpy.png"),
  caption: "Complete HARP output for the SAXPY kernel benchmark"
)
#v(5em)
#figure(
  image("../../figures/6-appendix/daxpy.png"),
  caption: "Complete HARP output for the DAXPY kernel benchmark"
)

#figure(
  image("../../figures/6-appendix/sgemm.png"),
  caption: "Complete HARP output for the SGEMM kernel benchmark"
)
#v(5em)
#figure(
  image("../../figures/6-appendix/dgemm.png"),
  caption: "Complete HARP output for the DGEMM kernel benchmark"
)

#figure(
  image("../../figures/6-appendix/reduce.png"),
  caption: "Complete HARP output for the reduction kernel benchmark"
)

#figure(
  image("../../figures/6-appendix/borrowing1.svg", ),
  caption: "Read-only borrowing visualization"
)<read_only>

#figure(
  image("../../figures/6-appendix/borrowing2.svg"),
  caption: "Read-write borrowing visualization"
)<read_write>