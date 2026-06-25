import sys
import mlir.ir as ir
from mlir.dialects.transform import interpreter

ctx = ir.Context()
# Parse a simple valid module
code = """
module {
  func.func @main() {
    return
  }
}
"""
module = ir.Module.parse(code, ctx)

# Create a syntactically correct transform sequence that will fail on empty op
transform_code = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match attributes {tag = "non_existent"} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled, %loops = transform.structured.tile_using_for %op tile_sizes [2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
"""

try:
    print("Parsing transform code...")
    t_module = ir.Module.parse(transform_code, ctx)
    print("Applying transform code...")
    interpreter.apply_named_sequence(module, t_module.body.operations[0], t_module)
except Exception as e:
    print("Caught transform exception in python:", e)

# Clear variables to trigger garbage collection/destruction
print("Cleaning up variables...")
del module
del t_module
del ctx

print("Exiting python cleanly...")
