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
