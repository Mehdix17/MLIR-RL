module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printI64(i64)
  func.func private @printF32(f32)
  func.func private @printNewline()
  func.func @main(%arg0: tensor<128x56x288xf32>, %arg1: tensor<1xf32>, %arg2: tensor<128x56x288xf32>) -> (tensor<128x56x280xf32>, i64) attributes {llvm.emit_c_interface} {
    %0 = call @nanoTime() : () -> i64
    %1 = linalg.pooling_nwc_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%arg0, %arg1 : tensor<128x56x288xf32>, tensor<1xf32>) outs(%arg2 : tensor<128x56x288xf32>) -> tensor<128x56x288xf32>
    %2 = bufferization.alloc_tensor() : tensor<3xf32>
    %3 = bufferization.alloc_tensor() : tensor<128x56x286xf32>
    %4 = linalg.pooling_ncw_sum {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%1, %2 : tensor<128x56x288xf32>, tensor<3xf32>) outs(%3 : tensor<128x56x286xf32>) -> tensor<128x56x286xf32>
    %5 = bufferization.alloc_tensor() : tensor<7xf32>
    %6 = bufferization.alloc_tensor() : tensor<128x56x280xf32>
    %7 = linalg.pooling_ncw_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%4, %5 : tensor<128x56x286xf32>, tensor<7xf32>) outs(%6 : tensor<128x56x280xf32>) -> tensor<128x56x280xf32>
    %8 = bufferization.alloc_tensor() : tensor<1xf32>
    %9 = bufferization.alloc_tensor() : tensor<128x56x280xf32>
    %10 = linalg.pooling_nwc_max {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%7, %8 : tensor<128x56x280xf32>, tensor<1xf32>) outs(%9 : tensor<128x56x280xf32>) -> tensor<128x56x280xf32>
    %11 = call @nanoTime() : () -> i64
    %12 = arith.subi %11, %0 : i64
    return %10, %12 : tensor<128x56x280xf32>, i64
  }
}
