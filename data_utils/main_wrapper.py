import re

def __remove_duplicate_args(args: list[str], shapes: list[str]):
    """Removes duplicate pairs from the list of paired arguments with shapes
    Args:
        args (list[str]): list of arguments
        shapes (list[str]): list of shapes

    Returns:
        list[str]: list of arguments without duplicates
        list[str]: list of shapes without duplicates
    """
    args_shapes = list(zip(args, shapes))
    seen = set()
    result = []
    for item in args_shapes:
        if item not in seen:
            seen.add(item)
            result.append(item)

    args = [x for (x, _) in result]
    shapes = [x for (_, x) in result]
    return args, shapes

def read_file_stream(filename: str):
    """used for huge files"""
    with open(filename, 'r') as file:
        for line in file:
            yield line

def extract_element_shape(shape: str) -> str:
    matches = re.findall(r'tensor<(?:\d+x)*([fi]\d+)>',shape)
    if matches == []:
        raise ValueError("shape is wrong")
        
    return matches[0]

def nn_transform_wrapper(operation: str) -> str:
    """ adds a main functions that allocates tensors for the nn model's arguments

    Args:
        operations (str): the functions signature for the nn model's forward function

    Returns:
        str: the main function to be inserted
    
    """
        
    fields = re.findall(r"\s*\(([^())]+)\)\s*->\s*([^(]+)",  operation)[0]
        
    args, shapes = [], []
    for f in fields[0].split(', '):
        arg, shape = f.split(':')
        shapes.append(shape.strip())
        args.append(arg)

    args = [arg.strip() for arg in args]
    shapes = [shape.strip() for shape in shapes]

    args,  shapes = __remove_duplicate_args(args,  shapes)

    shapes.append(fields[1])

    #############################################################
    dims = []

    for shape in shapes:
    
        if shape.startswith("tensor"):
            arg_dims = list(map(int,  re.findall(r'\d+',  shape[7:-5])))
            dims.append( arg_dims )
    
        else:
            dims.append( -1 )

    #############################################################
    func_name = re.search(r"@(\w+)",operation).group(1)

    func_call = f"func.call @{func_name}({', '.join(args)}) : ({', '.join(shapes[:-1])}) -> {shapes[-1]}"

    #############################################################
    # All code:
    code = ''

    code += "func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }\n"
    code += "func.func private @printI64(i64)\n"
    code += "func.func private @printNewline()\n"
    code += "\n"
    code += "\n"
    code += "func.func @main(){\n"
    code += "    %c1 = arith.constant 1: index\n"
    code += "    %c0 = arith.constant 0 : index\n"
    code += "    %n = arith.constant 2: index\n"
    code += "\n"
    code += "    %val_f32 = arith.constant 2.00000e+00 : f32\n"
    code += "    %val_i64 = arith.constant 2 : i64\n"
    code += "    %zero = arith.constant 0.00000e+00 : f32\n"
    code += "\n"
    for arg, shape, arg_dims in zip(args, shapes, dims):
        # print_info(arg,shape,arg_dims)
        if arg_dims != -1:
            tmp_arg = f'%tmp_{arg[1:]}'
            code +=f"    {tmp_arg} = bufferization.alloc_tensor() : {shape}\n"
            element_type = extract_element_shape(shape)
            code +=f"    {arg} = linalg.fill ins(%val_{element_type} : {element_type}) outs({tmp_arg} : {shape}) -> {shape}\n"
        else:
            code +=f"    {arg} = arith.constant 2.00000e+00 : f32\n"
    
    code += "        \n"
    code += "    scf.for %i = %c0 to %n step %c1 {\n"
    code += "        %t0 = func.call @nanoTime() : () -> (i64)\n"
    code += "        \n"
    code += f"         %outputmain = {func_call}\n"
    code += "        \n"
    code += "        %t = func.call @nanoTime() : () -> (i64)\n"
    code += "        %delta = arith.subi %t, %t0 : i64\n"
    code += "        func.call @printI64(%delta) : (i64) -> ()\n"
    code += "        func.call @printNewline() : () -> ()\n"
    code += "        \n"
    code += "    }\n"
    code += "    return\n"
    code += "}\n"

    return code
def nn_transform_wrapper_binding(operation: str) -> str:
    """ adds a main functions that allocates tensors for the nn model's arguments

    Args:
        operations (str): the functions signature for the nn model's forward function

    Returns:
        str: the main function to be inserted
    
    """
        #multiple return      \s*\(([^())]+)\)\s*->\s*\(([^(]+)\)
        #one return           \s*\(([^())]+)\)\s*->\s*([^(]+)
    fields = re.findall(r"\s*\(([^())]+)\)\s*->\s*\(([^(]+)\)",  operation)[0]
        
    args, shapes = [], []
    for f in fields[0].split(', '):
        arg, shape = f.split(':')
        shapes.append(shape.strip())
        args.append(arg)

    args = [arg.strip() for arg in args]
    shapes = [shape.strip() for shape in shapes]

    args,  shapes = __remove_duplicate_args(args,  shapes)

    shapes.append(fields[1])

    #############################################################
    num_return = len(fields[1].split(", ")) if "," in fields[1] else 0
        

    #############################################################
    func_name = re.search(r"@(\w+)",operation).group(1)

    # return_shape = shapes[-1]
    # shapes = shapes if len(shapes) == len(args) else shapes[:-1]

    func_call = f"func.call @{func_name}({', '.join(args)}) : ({', '.join(shapes[:-1])}) -> ({shapes[-1]})"

    #############################################################
    # All code:
    code = ""
    code += "func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }\n"
    code += "func.func private @printI64(i64)\n"
    code += "func.func private @printF32(f32)\n"
    code += "func.func private @printNewline()\n"
    code += "\n"

    code += f"func.func @main({', '.join([f'{arg}: {shape}' for arg,shape in zip(args, shapes) ])}) -> i64 attributes {{ llvm.emit_c_interface }} {{\n"
    
    code += "    %c1 = arith.constant 1: index\n"
    code += "    %c0 = arith.constant 0 : index\n"
    code += "    %n = arith.constant 2: index\n"
    code += "    %init_delta = arith.constant 0 : i64\n"
    
    code += " \n"
    code += "    %final_delta = scf.for %i = %c0 to %n step %c1 iter_args(%d = %init_delta) -> (i64) {\n"
    code += "    %t0 = func.call @nanoTime() : () -> (i64)\n"
    
    if not num_return:
        code += f"    %outputmain = {func_call}\n"
    else:
        code += f"    %outputmain:{num_return} = {func_call}\n"

    code += "    %t = func.call @nanoTime() : () -> (i64)\n"
    code += "    %delta = arith.subi %t, %t0 : i64\n"
    code += "    scf.yield %delta : i64\n"
    code += "}\n"
    code += "    return %final_delta : i64\n"
    code += "}\n"

    return code

def main_wrapper(filename,model_name, out):
    """Adds a main function to a model's code and writes the resulting code to the out file.

    The function reads the input file line by line, searches for the model function definition,
    and inserts a wrapper code after the function's return statement and closing brace.
    The wrapper code is generated using the `nn_transform_wrapper` function, which takes the
    function signature as input. The modified code is written to the specified output file.

    Args:
        filename (str): Path to the input file containing the model code.
        model_name (str): Name of the model function to be wrapped.
        out (str): Path to the output file where the modified code will be saved.
    """
    
    wrapped_code = ''
    return_found = False
    wrapped_code_added = False
    with open(out,"w") as o:
        for line in read_file_stream(filename):

            o.write(line)

            if not wrapped_code and model_name in line:
                signature = line[:-2].strip()

                wrapped_code = nn_transform_wrapper_binding(signature)
                # print(wrapped_code)

            if not return_found and "return" in line:
                return_found = True

            if not wrapped_code_added and wrapped_code and return_found and "}" in line:
                o.write(wrapped_code)
                wrapped_code_added = True
                continue

# TODO: Try to make it into a CLI script
if __name__ == "__main__":
    # filename = "../benchmarks/MobileNetV2_linalg_asm.mlir" 
    # filename = "../benchmarks/ResNet_linalg.mlir"
    # filename = "../benchmarks/VGG_linalg_asm.mlir"
    # filename = "./data_utils/tmp_models/EfficientNet_linalg.mlir"
    # filename = "./data_utils/tmp_models/ConvNeXt_linalg.mlir"
    
    # filename = "bart_linalg.mlir"
    # filename = "T5_linalg.mlir"
    filename = "gpt2_linalg.mlir"

    # filename = "./bert_linalg_1.mlir"
    # filename = "bart_linalg.mlir"
    # filename = "base-bert_linalg.mlir"
    # filename = 'graphsage_linalg.mlir'

    # model_name = "EfficientNet"
    # model_name = "ResNet"
    # model_name = "MobileNetV2"
    # model_name = "VGG" 
    # model_name = "forward"
    # model_name = "ConvNeXt"

    model_name = "main_graph"

    # out = f"./benchs/{model_name}-large-bench.mlir"
    # out = f"./benchs/bert-bench.mlir"
    # out = f"./benchs/bart-bench.mlir"
    # out = f"./benchs/T5-bench.mlir"
    out = f"./benchs/gpt2-bench.mlir"

    # out = "./benchs/base-bert-bench.mlir"
    # out = 'benchs/graphsage-bench.mlir'


    main_wrapper(filename, model_name, out)

    