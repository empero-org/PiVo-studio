// quick binary to verify weight loading
fn main() {
    let w = pivo_runner::onnx_loader::parse_onnx_initializers(
        &std::fs::read("../huggingface/onnx/encoder.onnx").unwrap()
    ).unwrap();
    for name in ["onnx::MatMul_1137","onnx::MatMul_1140","onnx::MatMul_1145","onnx::MatMul_1163","onnx::MatMul_1164","onnx::MatMul_1165","onnx::MatMul_1166"] {
        let t = &w[name];
        println!("{} {:?} {:?}", name, t.dims, &t.data[..4]);
    }
}
