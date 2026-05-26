## A Reinforcement Learning Environment for Automatic Code Optimization in the MLIR Compiler 

Mohammed Tirichine _New York University Abu Dhabi_ Abu Dhabi, UAE _ESI_ Algiers, Algeria tirichine.m@nyu.edu 

Nassim Ameur _New York University Abu Dhabi_ Abu Dhabi, UAE _ESI_ Algiers, Algeria na3758@nyu.edu 

Nazim Bendib _New York University Abu Dhabi_ Abu Dhabi, UAE _ESI_ Algiers, Algeria nb3891@nyu.edu 

Iheb Nassim Aouadj _New York University Abu Dhabi_ Abu Dhabi, UAE ia2280@nyu.edu 

Djad Bouchama _New York University Abu Dhabi_ Abu Dhabi, UAE _USTHB_ Algiers, Algeria db5394@nyu.edu 

Rafik Bouloudene _New York University Abu Dhabi_ Abu Dhabi, UAE _USTHB_ Algiers, Algeria rb5953@nyu.edu 

Riyadh Baghdadi _New York University Abu Dhabi_ Abu Dhabi, UAE baghdadi@nyu.edu 

_**Abstract**_ **—Code optimization is a crucial task that aims to enhance code performance. However, this process is often tedious and complex, highlighting the necessity for automatic code optimization techniques. Reinforcement Learning (RL) has emerged as a promising approach for tackling such complex optimization problems. In this project, we introduce MLIR RL, an RL environment for the MLIR compiler, dedicated to facilitating MLIR compiler research and enabling automatic code optimization. We propose a multi-discrete formulation of the action space where the action space is the Cartesian product of simpler action subspaces. We also propose a new method, called level pointers, to reduce the size of the action space related to the loop interchange transformation. This enables more efficient and effective learning of the policy. To demonstrate the effectiveness of MLIR RL, we train an RL agent to optimize MLIR Linalg code, targeting CPU. The code is generated from two domain-specific frameworks: deep-learning models generated from PyTorch, and LQCD (Lattice Quantum Chromodynamics) code generated from an LQCD compiler. The result of this work is a research environment that allows the community to experiment with novel ideas in RL-driven loop-nest optimization.** 

_**Index Terms**_ **—Automatic Code Optimization, Reinforcement Learning, MLIR, Deep Learning, Machine Learning, Compiler** 

## I. INTRODUCTION 

Writing high-performance code for compute-intensive applications is a challenging task that requires significant expertise. While manual code optimization can reduce execution time significantly, it is time-consuming and error-prone, making automatic compiler code optimization increasingly important. 

Several techniques have been proposed to enable automatic code optimization in compilers, including the use of integer linear programming (ILP) to find the best code optimizations [1], [2], and using tree-search methods guided by deep learning cost models [3]–[5]. 

More recent work has explored the use of Reinforcement Learning (RL) for automatic code optimization [6]–[11]. These approaches model the problem as sequential decisionmaking, where an RL agent selects a sequence of compiler transformations. While such approaches have made progress, they still have limitations: some are semi-automatic (requiring user-specified directives), others are implemented in research compilers with limited support for complex workloads and code transformations, and many target narrow domains (e.g., DNN graphs only), limiting generality. 

Making progress in this area requires the development of novel RL environments integrated within robust compilers that expose general, parameterized loop transformations across domains. In this context, the MLIR (Multi-Level Intermediate Representation) compiler [12] has emerged as a strong fit. It is an industrial-grade compiler infrastructure with a robust implementation and widespread adoption in industry and academia. Its multi-dialect architecture exposes a rich set of loop and data-layout transformations that are fine-grained and composable. Its ecosystem spans a wide range of front ends and back ends, enabling a single setup to cover diverse languages and hardware targets. These strengths make MLIR 

a strong foundation for research on RL-based automatic code optimization. However, to our knowledge, there is currently no RL environment for automatic code optimization natively integrated with MLIR. 

Although prior work, such as CompilerGym [13], provides an RL environment for LLVM, it is not directly usable in MLIR without substantial adaptation. In particular, building an MLIR RL environment requires solving many challenges. First, the action space modeling in CompilerGym is not suitable for automatic code optimization: CompilerGym models actions as selecting from a fixed set of LLVM passes (i.e., pass ordering over LLVM IR). In MLIR, loop-level optimization requires parameterized, location-specific transformations (e.g., choosing loops on which a transformation is applied, choosing tiling factors, fusion targets, etc.). Because CompilerGym does not expose such a structured, parameterized action space or enumerate legal application sites, MLIR optimization cannot be easily supported without effectively building a new environment. The second challenge when building an RL environment for automatic code optimization is the large size of the action space: each transformation can be applied at different loop levels and has multiple parameters, yielding a combinatorial number of choices. Third, implementing this action space is nontrivial: MLIR’s multi-dialect architecture exposes fine-grained, parameterized transformations, and building an action space that correctly applies these transformations requires significant engineering effort. The fourth challenge is the high cost of training an agent for the RL environment: evaluating long transformation sequences across many programs and repeatedly executing each variant to obtain stable measurements is computationally expensive, especially for large inputs. During development, this end-to-end training must be repeated many times to validate design choices, compare agents, and tune hyperparameters, making the cumulative training cost substantial. Such repeated training demands computing resources that are sometimes beyond the reach of the research community. These challenges make it difficult to simply adapt existing RL environments, motivating the need to build a specialized RL environment for automatic code optimization in MLIR. 

To address this gap, we propose MLIR RL, an RL environment for automatic loop-level code optimization in MLIR. We propose an RL agent with a multi-discrete action space: it first selects the transformation to apply and then its parameters. This is done by formulating the action space as the Cartesian product of much smaller subspaces. This enables the RL agent to learn a better policy and potentially find better sequences of optimizations. To further reduce the size of the action space, we propose a novel policy network that uses level pointers for the loop interchange transformation: instead of enumerating all the possible loop interchanges (which would be high), the network decides about the loop order, level by level, one at a time, significantly reducing the size of the action space and improving policy learning. 

To demonstrate the effectiveness of MLIR RL, we train an RL agent to optimize MLIR code generated from two 

domain-specific frameworks: deep-learning models generated from PyTorch, and LQCD (Lattice Quantum Chromodynamics) code generated from an LQCD compiler. LQCD is a branch of theoretical nuclear physics that uses large-scale simulations of quarks and gluons on a spacetime lattice to study the strong force and the structure of matter. MLIR RL specifically optimizes code in the Linalg MLIR dialect targeting CPU (Linalg is an MLIR IR dialect designed for linear algebra and deep learning computations). 

We evaluate MLIR RL against state-of-the-art frameworks (PyTorch [14], PyTorch compiler [15], Halide autoscheduler [16], and Halide RL [6]). The outcome of this work is a research infrastructure that the community can use to explore novel ideas in the topic of RL-based loop nest optimization. 

In summary, the contributions of this paper are: 

- 1) We propose MLIR RL, an RL environment for automatic code optimization in the MLIR compiler, facilitating further research in this area. 

- 2) We propose a multi-discrete action space and use level pointers to reduce the size of the action space and improve policy learning. 

- 3) We implement and evaluate our proposed approach, showing its effectiveness in optimizing MLIR code. 

- 4) We release MLIR RL to the community[1] . 

The rest of the paper is structured as follows: we begin with a background on MLIR and RL in compilers, followed by a detailed description of our RL environment, including the action space, states, and rewards. We then present the multidiscrete policy network and the use of level pointers. Finally, we evaluate MLIR RL on a set of benchmarks and compare it to four state-of-the-art compilers and frameworks: PyTorch, the PyTorch compiler, Halide autoscheduler, and Halide RL. 

## II. BACKGROUND AND RELATED WORK 

Many efforts in the compiler community have focused on automatic code optimization [1], [3]–[5], [9]–[11], [13], [17]– [45]. In this section, we provide an overview of relevant background and related work, focusing on MLIR and the use of Reinforcement Learning (RL) in compiler optimization. 

## _A. MLIR_ 

MLIR [12] is a framework designed to address the needs of modern compilers and heterogeneous hardware. It provides a flexible infrastructure for defining multiple levels of intermediate representations through various dialects, facilitating code translation and optimization across diverse domains. Notably, MLIR includes the Linalg dialect for linear algebra operations, the Affine dialect for expressing affine loops, and the Vector dialect for vectorization. MLIR also improves code optimization in DNN frameworks such as TensorFlow and PyTorch, offering a unified infrastructure that improves performance across frameworks. 

1https://github.com/Modern-Compilers-Lab/MLIR-RL 

## _B. Machine Learning for Compilers_ 

Machine learning has been used to improve compiler optimizations, notably to train a cost model that estimates the performance of optimized code. It was used in Tiramisu [5], [38], Halide [3], and TVM [46] to empower search algorithms such as beam search to efficiently find better schedules. Because these compilers rely on tree-search algorithms, they need to explore a large number of optimization candidates to find the best sequence of code optimizations. The Halide autoscheduler, for instance, explores millions of schedule candidates [3], which makes the process of code optimization slow. To constrain the search space, these compilers often impose restrictions, such as Tiramisu’s fixed exploration order [5]. Due to these limitations, recent efforts have shifted toward RL as a more scalable solution for automatic code optimization. 

## _C. Reinforcement Learning for Compilers_ 

More recent work has increasingly focused on RL due to its potential to train systems that automatically select the best sequence of actions, which is highly relevant for tasks such as loop optimization. For instance, Halide RL [6] employed RL to determine the best sequence of code optimizations and their parameters to minimize the execution time of image processing pipelines. Halide RL is not fully automatic, though, as it requires an initial input set of directives provided by a user for the RL agent to select from. This is different from our proposed RL environment, where the whole optimization process is automated. 

Other work [9], [10] utilized RL to target the problem of phase ordering. It aims to automate high-level synthesis (HLS) by selecting the best order of compiler passes. Our proposed RL approach, rather than relying on passes, tackles the task of code optimization by selecting the optimizations to apply, their parameters, in which order to apply them, and on which part of the code. In addition, AutoPhase [10] targets HLS and does not target CPUs, which we focus on. 

Prior works like Chameleon [7], REGAL [47], and X- RLflow [48] leverage RL to accelerate DNN graphs, focusing on high-level tasks such as parameter tuning for convolutions, model parallelism, or graph rewriting. In contrast, our method operates at a lower level (MLIR linear algebra dialect) rather than on deep learning graphs. This allows our RL environment to target multiple domains, which we demonstrate using both deep learning models and LQCD computations. 

SuperSonic [11] introduces a meta-optimizer to search and tune RL architectures, providing a tool for the automatic design of RL environments. It addresses a problem that is orthogonal to our contribution, and our RL environment can also benefit from these techniques. 

To facilitate research, CompilerGym [13] provides RL environments for tasks like LLVM phase ordering and GCC flag selection. Similarly, PolyGym [37] leverages polyhedral optimization for general-purpose computation. 

While these two environments are important milestones towards democratizing research on RL in compilers, they are not designed for automatic code optimization in the MLIR 

compiler. Therefore, one needs to spend a significant effort to fully integrate them into MLIR before being able to use them in their research. Moreover, adapting them to the task of automatic code optimization in MLIR is tedious. This is mainly because developing an effective action space that has a comprehensive list of optimizations in the MLIR compiler requires effort and expertise. Most of our effort in building our proposed RL environment was spent on building this effective action space. 

The MLIR compiler is now widely used by the research community and is becoming the backbone of several deep learning frameworks and domain-specific language compilers. Therefore, we believe that a specialized RL environment specifically designed for automatic code optimization in MLIR is needed. 

## _D. MLIR-based Compilers for Machine Learning_ 

MLIR has become the foundation of several modern compiler stacks targeting machine learning workloads. The principal aim is to create a unified system that can take models from multiple frameworks (PyTorch, TensorFlow, ONNX, etc.) and progressively lower them using MLIR, enabling optimization at different levels of abstraction. These systems differ in how they use MLIR, which dialects they optimize, and how optimization decisions are made. 

_1) IREE:_ IREE [49] is an end-to-end compiler and runtime system designed to lower machine learning models to a unified intermediate representation. It relies heavily on the _Linalg_ dialect as its core abstraction for compute-intensive operations. IREE’s compilation flow lowers high-level inputs (from TensorFlow, JAX, PyTorch, etc.) into _Linalg_ operations, where it performs hardware-agnostic transformations such as fusion and tiling. These transformations are typically applied via a predefined pass pipeline that utilizes static heuristics and target-specific configurations to determine tile sizes and vectorization factors. The code is eventually lowered to the HAL (Hardware Abstraction Layer) dialect for execution on diverse backends including CPUs, GPUs, and micro-controllers. 

_2) ONNX-MLIR:_ ONNX-MLIR [50] is a compiler specifically designed to compile valid Open Neural Network Exchange ( _ONNX_ ) graphs into optimized binary code. It introduces two specific dialects: the _ONNX_ dialect, which maps oneto-one with the _ONNX_ standard specification, and the _Krnl_ dialect, which serves as a bridge for loop-level optimizations. Optimizations are applied at the _Krnl_ dialect level. The _Krnl_ dialect allows for the explicit representation of schedules enabling transformations such as tile, skew, and transpose. Optimization decisions, such as loop ordering and tile sizing, are generally driven by static analysis and lowering rules implemented within the compiler’s conversion passes. 

_3) XLA:_ XLA [51] is a domain-specific compiler originally built for TensorFlow but now supporting JAX and PyTorch. XLA implementations increasingly leverage MLIR infrastructure, particularly through the MHLO (MLIR HLO) and StableHLO dialects, which represent high-level linear algebra operations. XLA’s primary optimization strategy 

revolves around aggressive kernel fusion to reduce memory bandwidth usage, along with layout assignment and buffer analysis. The compiler selects optimizations using analytical cost models and greedy heuristics to decide which operations to fuse or how to layout memory. 

While IREE, ONNX-MLIR, and XLA are robust production compilers, they primarily rely on static heuristics and predefined pass pipelines to select optimizations. For instance, IREE uses target configurations to guide tiling, and XLA uses a cost model for fusion. In contrast, MLIR RL replaces these fixed heuristics with a learned policy. By exposing the _Linalg_ dialect transformations (tiling, fusion, interchange) as a multi-discrete action space, MLIR RL allows a Reinforcement Learning agent to explore and discover optimization schedules automatically. Unlike ONNX-MLIR’s _Krnl_ dialect which requires explicit scheduling logic, or XLA’s greedy fusion, our system enables the agent to learn which sequence of transformations to apply and how, based on empirical feedback, potentially uncovering optimization sequences that static heuristics might miss. 

## III. OVERVIEW OF MLIR RL 

Our RL framework uses the actor-critic RL architecture and is defined by the following key components: 1) _Action space (detailed in Sec. IV-A):_ defines the set of code optimizations that the agent can apply to the program being optimized; 2) _State representation (detailed in Sec. IV-B):_ provides a representation of the program being optimized; 3) _Reward model (Sec. IV-C):_ provides a function that scores the quality of actions, guiding the agent toward actions that produce faster code; 4) _Actor-critic networks (Sec. V):_ the actor selects the next code optimization to apply given the current program state, and selects its parameters, while the critic estimates the expected return and provides the learning signal for policy optimization during training (the critic is only used during training to train the policy). 

We train our proposed RL agent on _a dataset_ that we describe in Sec. VI, using the _PPO learning algorithm_ . 

In this project, we focus on optimizing operations expressed in the Linalg MLIR dialect. The Linalg MLIR dialect provides a structured IR for linear-algebra and deep learning operations (e.g., matmul, convolutions, and generic elementwise and reduction kernels), defined over tensors or buffers with explicit iteration spaces and affine indexing maps. We chose the Linalg dialect because of its robustness, maturity, and wide set of code transformations that it supports. Early in the project, we considered other dialects, such as the Affine and the SCF dialects, which operate at a lower level (loop nests), but we faced difficulties due to the limited number of transformations that are well supported in those dialects. In addition, the main domains that we want to optimize (deep learning and LQCD computations) can both be fully expressed in this dialect. Listing 1 shows an example of a matrix multiplication implemented using the Linalg generic operation. 

Although in this work we focus on optimizing the Linalg dialect, the design of MLIR RL is dialect-independent, in 

principle. The environment, reward interface, state representation, and RL infrastructure are fully reusable across MLIR dialects. Extending the system to other dialects (e.g., Tosa or Affine) mainly requires adapting the action space and the feature-extraction module, rather than redesigning the full framework. More precisely, our framework relies on the following assumptions: 1) the IR representation is at the loop level; 2) computations are sequences of loop nests (for-loops) manipulating arrays; 3) the dialect exposes a set of high-level loop-nest transformations; 4) the dialect provides a mechanism to guarantee the correctness of transformations, either because transformations are correct by construction or because legality is checked after dependence analysis. A dialect that satisfies these assumptions would, in principle, be compatible with MLIR RL, although demonstrating the effectiveness of our framework in those cases is left for future work. 

Listing 1: MLIR linalg.generic operation 

linalg.generic { indexing_maps = [ **affine_map** <(d0,d1,d2) -> (d0,d2)>, **affine_map** <(d0,d1,d2) -> (d2,d1)>, **affine_map** <(d0,d1,d2) -> (d0,d1)> ], iterator_types = ["parallel", "parallel", "reduction"] } **ins** (%A, %B: memref<256x1024xf32>, memref<1024x512xf32>) **outs** (%C: memref<256x512xf32>) { ˆbb0(%a: f32, %b: f32, %c: f32): %d = arith.mulf %a, %b : f32 %e = arith.addf %c, %d : f32 linalg.yield %e : f32 } 

Since code usually contains multiple operations, MLIR RL processes one operation at a time. Operations are traversed in reverse order (from consumers to producers), as the Linalg fusion transformation has limited ability to fuse a modified producer. Starting from the consumer thus preserves more fusion opportunities. When multiple producer operations exist for a given consumer operation, we select the last producer (the one that occurs in the code right before the consumer, textually) as the next to fuse with the current consumer being optimized. 

## IV. REINFORCEMENT LEARNING ENVIRONMENT 

In this section, we detail the components of our proposed RL environment. Specifically, we discuss the action space of the environment, the state and observations, and the proposed reward functions. We will use the notation presented in Table I throughout the rest of the paper. 

||**Symbol**<br>_N_<br>_M_|**Description**<br>Number of loop levels in a loop nest<br>Number of tile sizes|
|---|---|---|
||_D_|Rank of matrix accesses|
||_L_|Number of accessed matrices|
||_τ_|Length of a transformation sequence|



TABLE I: Notation used in the paper. 

## _A. Action Space_ 

An action _at ∈ A_ , where _A_ is the action space, allows the agent to transition from a state _st_ to another state _st_ +1. In this 

project, an action is a code transformation (code optimization) that can be applied to a loop nest (an operation). We focus on the following transformations: 

- **Tiling:** This transformation allows the working set of data to fit better into the cache, thereby improving memory access patterns and overall performance. We use the notation _T_ ( _t_ 1 _, t_ 2 _, . . . , tN_ ) to specify that tile size _ti_ is used to tile the loop level _i_ . Note that a tile size of zero indicates no tiling. 

- **Tiled Parallelization:** This transformation in the Linalg dialect is the combination of two transformations: tiling followed by the parallelization of the outermost loop. One can apply parallelization alone, without tiling, by selecting a tile size of 1 for all the loop levels. Parallel execution is achieved by generating _scf.parallel_ operations, which are lowered to the _OpenMP_ dialect (specifically _omp.wsloop_ ) and subsequently to _LLVM IR_ , inserting the necessary runtime calls to the _OpenMP_ library to manage thread creation and synchronization. 

- **Tiled Fusion:** This transformation is the combination of tiling followed by a fusion. It merges a producer loop, which generates data in the form of a tensor, with a consumer loop that accesses that data. This serves to reduce the need for storing intermediate results, which can improve data locality. In Linalg, a loop must be tiled first, before being fused. This is because fusion is implemented at the tile granularity of the consumer. One first tiles the consumer, which creates explicit outer loops over tiles. Only then can the producer be cloned and moved inside those outer tile loops so each tile computes its needed portion locally. 

- **Interchange:** Interchange involves permuting the order of loop levels using a permutation denoted by _I_ ( _a_ 1 _, a_ 2 _, . . . , an_ ), where _ai_ represents the new order (index) of loop _i_ . For example, if we have a loop nest with 3 loop levels, the interchange _I_ (2 _,_ 0 _,_ 1) will move the innermost loop to become the outermost. 

- **Vectorization:** vectorizes the innermost loop nest. 

- **No Transformation:** A special action that allows the agent to stop optimizing the current operator being optimized (current consumer being optimized) and move to the next operator (one of the producers of this consumer operator). When there is no producer left to optimize, the agent stops. 

To illustrate the effect of these optimizations on MLIR code, Listing 2 shows the result of applying _Tiled Parallelization_ with tile sizes [8 _,_ 8 _,_ 0], followed by _Vectorization_ , to the operation in Listing 1. 

Each action is defined by a transformation and an associated set of parameters (if required). The sizes of the action space for each transformation are as follows: 

- **Tiled transformations (** _**Tiling**_ **,** _**Tiled Parallelization**_ **,** _**Tiled Fusion**_ **):** _M[N]_ . 

- **Interchange:** _N_ !, representing the number of possible loop permutations. 

- **Vectorization and No Transformation:** 1, since no additional parameters are required. 

The total size of the action space _|A|_ is: _|A|_ = 3 _· M[N]_ + _N_ !+2 _1) Multi-Discrete Formulation:_ The above action space is large, and learning a policy for such a large action space is not trivial. To address this, we reformulate the above action space into a _multi-discrete action space_ , where an action is represented as the Cartesian product of multiple subaction spaces (i.e., an action is the combination of multiple simpler sub-actions). As a result, the action space becomes a composition of smaller discrete distributions defined as follows: 

- **Transformation Selection:** A categorical distribution over six possible transformation options (Tiling, Tiled Parallelization, etc.). 

- **Tiling Sizes for each Tiled Transformation:** The goal is to select a tile size for each loop level; therefore, we define _N_ categorical distributions (one per loop level), each over _M_ candidate tile sizes. 

- **Interchange:** We propose two formulations for interchange: 

   - **Enumerated Candidates:** We enumerate a subset of loop interchange candidates by swapping two loop levels that are either adjacent or separated by one or two levels. This restricted enumeration aims to keep the action space tractable, substantially reducing the number of possibilities while still capturing useful transformations. 

   - **Level Pointers:** Inspired by the work of Vinyals et al. [52], the interchange action is decomposed into a sequence of sub-steps. At each step _i ∈_ [0 _, N_ ), the agent selects a loop _n ∈_ [0 _, N_ ) to be placed at position _i_ . This requires a probability distribution over _N_ loops. This method manages to cover all possible permutations, without the need to enumerate all the possibilities. 

_2) Action Mask:_ Not all actions are valid at every time step. To address this, we provide the agent with an _action mask_ that filters out invalid actions. The mask is created based on the current state. An example of actions that need to be masked is the vectorization action when the innermost loop is too large (having more than 512 iterations). In such a case, the MLIR vectorization pass leads to the generation of large and inefficient code, as the vectorization pass in MLIR fully unrolls the inner loop. This causes excessive memory consumption and significantly slows down the RL training. 

## _B. States and Observations_ 

A linalg code is composed of a sequence of linear algebra operations and loop nests (implemented using the _generic_ linalg operator). Each operation and loop nest in the code is represented with a vector of features. Each vector is the result of concatenating a set of features (shown in Figure 1). We present these features in this section. 

- **Operation Type:** A one-hot vector encoding the Linalg operation type. The considered types include: matmul, 

1) MLIR Linalg Operation 

**==> picture [443 x 163] intentionally omitted <==**

**----- Start of picture text -----**<br>
linalg.matmul ins(%A, %B: tensor<256x1024xf32>, tensor<1024x512xf32>) outs(tensor<256x512xf32>) -> tensor<256x512xf32> 1) MLIR Linalg Operation<br>Operation Type Loop Ranges Vectorization Pre-conditions Indexing Maps Operations Count<br>Matmul Upper Bounds Iterator Types True (d0, d1, d2) -> (d0, d2) mul (*):    1 2) Parse Code<br>    256        512   parallel       parallel (d0, d1, d2) -> (d2, d1)(d0, d1, d2) -> (d0, d1) add (+):    1<br>            1024            reduction<br>generic 0<br>matmul 1 256 1 111 000 000 dim 1 +- 10<br>conv 0 512 1 1 000 111 000 dim 2 * 1 3) Extract Features<br>poolingadd 00 1024 0 map 1map 2map 3 000 000 000 dim 3 / 0<br>other 0 normalize d0 d1 d2 exp 0<br>Action History<br>Tiled Transformations Interchange Representation Vector 4) Concatenate features<br>nb_steps x nb_loops x nb_tiles nb_steps x nb_loops x nb_loops<br>**----- End of picture text -----**<br>


Fig. 1: The pipeline of extracting the features from a Linalg operation and building the representation vector. 

conv 2d, add, pooling, and generic, with an additional unknown category to account for any other operation type that was not seen during training and that the system may encounter. Generic operations in the linalg dialect are used to express general loop nests. Note that some common operations such as _ReLU_ don’t exist in the Linalg dialect so they are explicitly coded using linalg.generic which means that their type here is also considered _generic_ . Using named operations, when available, helps the agent learn optimization patterns that are specific to particular loop-nest structures, which can lead to better optimization decisions than treating all operations as fully generic. 

- **Loop Ranges:** We extract two key characteristics of each loop: the upper bound and the iterator type. The lower bound and step are omitted, as they are always fixed to 0 and 1, respectively, in Linalg operations. The iterator type (reduction or parallel iterator) is essential for determining the legality of parallelization, depending on whether the iterator is used for a reduction ( _reduction_ ) or not ( _parallel_ ). 

- **Vectorization Pre-conditions:** In MLIR, not all Linalg operations are eligible for vectorization, as specific conditions must be satisfied beforehand. If these conditions are not met, vectorization attempts will fail. To account for this, we include a boolean flag indicating the outcome of these pre-condition checks, thereby informing the agent whether vectorization is feasible for a given operation. 

- **Indexing Maps:** Each Linalg operation is associated with indexing maps that specify how data is read or written in tensors. An indexing map defines the mapping between loop iterators ( _d_ 0 _, d_ 1 _, . . . , dN_ ) and the indices of the accessed tensor, expressed as an affine function of the iterators for each dimension. For example, the map (d0, d1, d2) -> (d0 + 1, 3 * d2) indicates that the operation has three iterators (three loop levels), where the first tensor dimension is accessed at _d_ 0 + 1 and the second at 3 _· d_ 2. Inspired by work done by Baghdadi et al. [5], we represent each map as a polyhedral access matrix of size _D × N_ , where each row corresponds to one dimension of the tensor, and each column corresponds 

to a loop iterator. The entries of the matrix capture the coefficients of each loop iterator. 

**==> picture [199 x 49] intentionally omitted <==**

**----- Start of picture text -----**<br>
array [d0, d0 + 2 * d1 - 3 * d2, 1 - d1] 1 0 0 dim 1<br>1 2 -3 dim 2<br>0 -1 0 dim 3<br>d0 d1 d2<br>**----- End of picture text -----**<br>


Fig. 2: Example of an access matrix. 

- **Operations Count:** We count the number of each arithmetic operation, including: addition (+), subtraction (-), multiplication (*), division (/), and exponential (exp). We store these numbers in a vector that represents operation counts. 

- **Action History:** We record the sequence of applied transformations by storing their parameters (when they exist) in one-hot encoded matrices, indexed by time. At each time step, if a transformation is not applied, the corresponding entry is zero; otherwise, it stores the chosen parameters. More details on the exact encoding for each transformation type are provided in Appendix A. 

## _C. Reward_ 

An intuitive reward for the task of code optimization is the speedup of the optimized code (acceleration rate), which is the ratio between the old execution time to the new one. However, since the goal of reinforcement learning is to maximize cumulative rewards, we chose to use the logarithm of the speedup. This approach leverages the additive property of logarithms, making it more suitable for reward accumulation. 

The reward is given at the end of an episode. We assign a reward of 0 after every step except for the last step, where we execute the optimized code and return the speedup. 

We have also tried another reward structure where intermediate rewards are provided after each step, in addition to the above terminal reward, but the use of such intermediate steps does not improve the policy yet it makes the training slow (because we need to execute the optimized code after each step, to calculate the reward). 

**==> picture [435 x 132] intentionally omitted <==**

**----- Start of picture text -----**<br>
Transformation Selection<br>Producer<br>Representation LSTM<br>Vector Tiling<br>Backbone Tiled Parallelization<br>Tiled Fusion<br>RepresentationVector LSTM<br>LSTM Embedding<br>Embedding<br>Interchange<br>**----- End of picture text -----**<br>


Fig. 3: The RL agent’s policy network architecture consists of a backbone that processes the input representation vector into a feature vector that is then passed to the subnetworks to predict the transformation to apply and its parameters. 

**==> picture [455 x 145] intentionally omitted <==**

**----- Start of picture text -----**<br>
No Transformation<br>Tiling<br>Tiled Parallelization T 11 T 12 ... T 1M T 1<br>T 21 T 22 ... T 2M T 2<br>Tiled Fusion T N1 T N2 ... T NM T N<br>Interchange<br>Vectorization<br>LSTM Embedding Embedding Embedding<br>Embedding<br>a) b) c)<br>ReLU ReLU ReLU<br>Dense (6)<br>Dense (512) Dense (512) Dense (512) Dense (NxM)<br>**----- End of picture text -----**<br>


Fig. 4: The detailed architecture of the networks used in the policy network: a) The backbone; b) The transformation selection network; c) Tiled transformations network. The interchange network varies in size depending on the interchange method, but it is always a dense layer outputting one distribution. 

## V. ACTOR-CRITIC NETWORK ARCHITECTURE 

A central component of Reinforcement Learning is the _agent_ , the entity responsible for selecting actions given input states. In this section, we describe our agent, designed for the previously introduced MLIR environment. We adopt an actor-critic architecture, which decomposes the agent into two sub-components: the _actor_ , responsible for learning a policy to select actions, and the _critic_ , responsible for evaluating how good a state is under the actor’s policy. 

The actor is the policy network, which learns a policy _π_ that maps each state _s ∈ S_ to a probability distribution over actions. Formally, _π_ ( _a|s_ ) denotes the probability of taking action _a ∈ A_ when in state _s_ . 

The critic is the value network, which estimates the value function _vπ_ ( _s_ ), defined as the expected return when starting from state _s_ and following policy _π_ . 

## _A. Policy Network_ 

Since the action space is partitioned into multiple subspaces, the policy network must sample a sub-action from the subspaces to construct the final action. We first sample the type of the transformation to apply (one of 6 transformation options), then we sample the parameters of the selected transformation. To implement this, we use a single policy network that has three components: a network to extract an embedding for the 

code being optimized, a backbone to create a rich embedding from the input embedding, and multiple heads to map the rich embedding to sub-actions (one head for selecting the transformation, and multiple other heads for selecting the parameters of transformations, where each head is specialized in selecting the parameters of a given transformation). 

More precisely, the policy network is composed of the following three components: 

- 1) **Producer-Consumer Embedding:** The goal of this component is to take the code representation as input and produce an embedding for that code. Instead of creating an embedding for the whole code, which would require passing a large input to the model, making learning hard, we extract the representation of two Linalg operators (a consumer and its producer). The goal is to make the input size smaller, which would simplify learning. Passing the representation of two operators is sufficient because we optimize code operator by operator, starting from the last consumer in the code. While optimizing this single operator, we decide whether to fuse it with its producers. Therefore, at a given time, we only need to pass the representation of two operators. To create an embedding that captures the relation between the producer and the consumer, we feed their representation vectors sequentially into an LSTM layer with 512 units. 

The final hidden state of the LSTM is used as the producer-consumer embedding. Although our preliminary tests showed that a dense network yields similar results, we selected an LSTM architecture because it naturally supports future extensions towards multi-producer fusion. 

- 2) **Backbone Network:** The LSTM embedding is passed through a backbone consisting of three fully connected layers with 512 neurons each, activated by ReLU [53] (see Figure 4.a). 

- 3) **Action Heads:** The embedding created by the backbone is fed into five different output heads: 

   - **Transformation Selection:** A fully connected layer of size 6 (corresponding to the number of transformation options), followed by a Softmax activation [53], which produces a probability distribution over the transformation options (see Figure 4.b). 

   - **Tiled Transformations:** Three heads, each consisting of a fully connected layer of size _N × M_ , reshaped into a 2-D matrix. Each row of the matrix is independently activated with Softmax, producing one distribution per loop level, each over _M_ candidates, to select a tile size for each loop (see Figure 4.c). 

   - **Interchange:** A fully connected layer followed by Softmax. The size of this layer depends on the chosen interchange method: 3 _N −_ 6 for enumerated candidates, or _N_ for level pointers (see Section IV-A1 for definition and Appendix B for a detailed explanation). 

## _B. Value Network_ 

The value network also uses three components. The first two components are identical to the first two components of the policy network (the producer-consumer embedding and the backbone network). The backbone embedding is then passed through a fully connected layer with a single output neuron, which estimates the state-value function _vπ_ ( _s_ ). 

## VI. DATASET 

Our goal is to train our agent to optimize code from two domains: deep learning and LQCD computations. Therefore, we created a dataset composed of code from these two domains. The total size of the dataset is 3959 training examples. The rest of the section describes how we created the dataset. 

## _A. Deep Learning Data_ 

We curated a new training dataset specifically for our task. Since our objective is to optimize full neural network code, we required a dataset composed of two types of computations: single deep learning operators and sequences of deep learning operations similar to those found in real neural network models. To construct this dataset, we adopted two approaches. First, we collected commonly used deep learning operators from open source models; Second, we randomly synthesized sequences of deep learning operations. To keep the training computationally feasible, we limited the sequence lengths to L=5, which provides a balance between keeping the training time reasonable while still allowing the agent to learn how 

to handle multiple operations simultaneously. Note that the agent, while optimizing a sequence of operators, only represents (perceives) two operators at a time and does not represent (perceive) the whole sequence of operators. 

||**Operation**|**Training set**|
|---|---|---|
||Matrix multiplication|187|
||2d convolution|278|
||Maxpooling|250|
||Matrix addition|271|
||ReLU|149|
||Total|1135|



TABLE II: The distribution of each DNN operator in the singleoperator training sets. 

To generate the dataset of single operators, we collected 121 state-of-the-art models from TensorFlow Hub and Hugging Face, spanning from vision models to transformers. We then collected the operations used in these models (e.g., conv, relu, matmul, etc.). From the collected operations, we selected the most frequently occurring ones. For each of these operations, we generated many variants by varying the input sizes and shapes. Each operation variant became a training example in our training dataset. We collected a total of 1135 training examples. These operations span multiple types, including matrix multiplication, convolutions, max pooling, additions, and ReLU activation (more details about the composition of this dataset in Table II). 

To generate the random operator sequences, we generated a random sequence of length L=5. Each operation in the sequence takes as input the output of the previous operation. The operations are generated with random shapes and are randomly selected from the following set of operations: add, matmul, relu, conv_2d and pooling, sigmoid, and softmax_2d. 

## _B. LQCD Data_ 

We integrated MLIR RL as a backend for a DSL compiler developed for the LQCD domain (this compiler is still unpublished). LQCD is a method for studying the strong force by simulating quarks and gluons on a space-time grid. A common task in LQCD is to compute _correlators_ , which are mathematical objects used to study the properties of particles and their interactions. LQCD computations are usually a long sequence of loop nests that read and write to tensors. Loop nests are usually deep (reaching more than 12 loop levels), and some of the accesses to tensors are irregular. Many of the loop nests are parallel, and many of them contain reductions at the inner levels. 

The LQCD compiler takes as input a DSL code that is embedded in Python. The LQCD compiler then lowers this code and generates Halide code as output. We have built a pass to translate Halide code to the MLIR Linalg dialect. We then use MLIR RL to optimize it. 

The LQCD compiler has 7 tests, each test is a large LQCD code (thousands of lines) designed to test the ability of the compiler to compile patterns that are commonly found in 

LQCD codes. We extracted all the loop nests from these tests. For each of these loop nests, we generated many variants by varying the input sizes. Each loop nest variant became a training example in our training dataset. In total, we collected 691 training examples. 

## VII. EXPERIMENTS AND RESULTS 

## _A. Experimental Setup_ 

_1) Hardware and Software Configurations:_ We perform our evaluations on a multicore dual-socket node, each socket is a 14-core Intel(R) Xeon(R) CPU E5-2680 v4 @ 2 _._ 40 GHz with 64 GB RAM total. We use 16 nodes identical to this one for the training. The transformations are implemented using MLIR, built on LLVM release 19. 

_2) Benchmarks:_ We evaluate MLIR RL on three sets of benchmarks: 

- _Deep Learning Operators_ : we use a benchmark of common operations in neural networks, namely matrix multiplication, convolution, pooling, addition, and ReLU. We use input data sizes and shapes that we obtained from widely used models (e.g., ResNet). The data sizes and shapes used for the evaluation were not seen during training. 

- _Neural Network Models_ : we also evaluated MLIR RL on a set of 3 neural network models: ResNet-18, VGG, and MobileNetV2. We use the PyTorch implementations of these models, and automatically generate the equivalent MLIR Linalg code using Torch-MLIR [54]. We provide more details about these models and their composition in Appendix C. 

- _LQCD Applications_ : This benchmark suite includes three increasingly complex applications: 1) _Dibaryon–Dibaryon:_ A baryon is a three-quark bound state (e.g., proton or neutron). A dibaryon is two baryons bound together (six quarks total). Thus, this benchmark studies interactions between two dibaryons (i.e., two six-quark states); 2) _Dibaryon–Hexaquark:_ comparing a two-baryon system with a six-quark exotic particle; and 3) _Hexaquark–Hexaquark:_ the heaviest case with correlators between two six-quark states. These LQCD applications have a number of lines ranging from _1000_ to _8000_ lines of MLIR Linalg code. 

_3) Metrics:_ We report the speedup of the optimized codes over the baselines. Our baselines are the MLIR implementations of the benchmarks compiled with the MLIR compiler without loop-level optimizations and with _O3_ applied automatically by LLVM (we use the same MLIR compiler pipeline that we use in MLIR RL, but we disable the selection of loop-level code optimizations done by MLIR RL). We run each code 5 times, and take the median of its _execution times_ . A speedup higher than 1 means that the optimized code is faster than the baseline unoptimized code. 

_4) Comparison with state-of-the-art:_ We compare MLIR RL to the following state-of-the-art compilers and frameworks. 

- _PyTorch_ [14]: this is the PyTorch framework version 

   - 2.8.0 used with the Intel MKL DNN library (v3.7.1). 

- We evaluate PyTorch on the deep learning operators and deep learning models. We perform 10 warm-up executions, followed by 11 measurement runs. The execution time is recorded in each measurement run, and we report the median. 

- _PyTorch compiler_ [15]: a state-of-the-art industrial compiler for PyTorch. We evaluate the PyTorch compiler on the deep learning operators and deep learning models. The PyTorch compiler is invoked using _torch.jit.script_ , and we follow the same evaluation protocol as for PyTorch (warm-up followed by measurement runs). 

- _Halide RL_ [6]: a reinforcement learning framework for the Halide compiler. We evaluate Halide RL on the deep learning operators. We do not evaluate it on full deep learning models or LQCD applications because its design makes supporting these cases difficult, and their system currently lacks automated tools to support these sources. 

- _Halide autoscheduler_ [16]: an autoscheduler for the Halide compiler, used as the default autoscheduler for optimizing the original LQCD applications. We evaluate it on the LQCD applications. 

_5) RL Agent Training:_ We train the agent for 10,000 steps (each step includes trajectory collection and 4 PPO update epochs). Training takes approximately 5 days and 7 hours on 16 CPU nodes (node characteristics described above). 

We set the maximum number of loop levels in a loop nest to 12, the number of tile sizes to 8 (including zero to represent no tiling), the maximum number of arrays accessed in each loop nest to 14, the maximum rank of array accesses to 12, and the maximum schedule length to 5. 

The RL agent is trained using Proximal Policy Optimization (PPO) [55] with a learning rate of 0.001 and a clip range of 0.2 to ensure stable policy updates. We set the discount factor to _γ_ = 1 _._ 0 because rewards are delayed until the end of the trajectory, avoiding excessive discounting for long codes. The GAE parameter was set to _λ_ = 0 _._ 95 to balance bias and variance in advantage estimation. Each trajectory corresponds to the full transformation sequence of all operations in a code sample. We collect trajectories from 64 code samples at a time, split the data into mini-batches of size 32, and perform 4 PPO update epochs. The value loss coefficient is set to 0.5 and the entropy coefficient to 0.01. 

## _B. Overhead of the Compilation Pass_ 

The overhead introduced by our compilation pass primarily comes from two sources: calls to the policy network and the application of MLIR transformations selected by the policy. The policy network is queried multiple times per program in order to generate the full optimization sequence. The average total inference time (CPU Only) is 0.028 _s per code sample_ (measured over single deep-learning operators and LQCD applications). Applying the selected transformations also incurs cost: for single deep-learning operators, MLIR takes on average 0.089 _s per code sample_ to apply the full sequence of transformations, while for LQCD applications the transformation time is 0.8 _s per code sample_ . 

## _C. Evaluation Results_ 

In this section, we present the results of our evaluation. We first present the results of our evaluation on the benchmark of deep learning operators, then our results on the deep learning models, and then on the LQCD applications. 

_1) Evaluation on the Deep Learning Operators:_ In this section, we show the evaluation of MLIR RL on the benchmarks of single deep learning operators. We compare the performance of MLIR RL to PyTorch [14], PyTorch compiler [15] and Halide RL [6]. We present the speedups of code generated by these compilers over unoptimized MLIR code (baseline). 

Figure 5 shows the results. On average, the relative performance varies significantly across operator types. For _Add_ and _ReLU_ , MLIR RL achieves competitive speedups compared to both PyTorch and the PyTorch compiler, showing that the agent can reliably learn effective schedules for elementwise operators whose performance is primarily bounded by memory throughput. For _Maxpooling_ , MLIR RL actually outperforms both PyTorch and the PyTorch compiler, achieving on average 3 _._ 3 _×_ higher speedups, mainly because MLIR RL automatically discovers effective tiling sizes that reduce memory traffic in these stencil-like operators for the target CPU. 

For _Conv2D_ and _Matmul_ , however, MLIR RL generates code that is slower than PyTorch and PyTorch compiler (on average 2 _._ 16 _×_ slower on _Matmul_ and 6 _._ 71 _×_ slower on _Conv2D_ ). This is because they leverage architecture-specialized kernels, register tiling, and aggressive vectorization implemented in libraries such as oneDNN. Although MLIR provides these capabilities, our current action space does not expose them yet, limiting the ability of the agent to optimize these compute-intensive programs. 

Compared to Halide RL, MLIR RL achieves competitive results on _Add_ and _ReLU_ , but for _Maxpooling_ , Halide RL slightly outperforms MLIR RL on all sizes (1 _._ 25 _×_ on average) due to the inability of our system to vectorize these operations. As for _Matmul_ , MLIR RL consistently achieves better speedups than Halide RL with an average of 5 _._ 32 _×_ indicating that MLIR RL manages to perform effective tilings and vectorization. 

_2) Evaluation on the Deep Learning Models:_ The following table presents the speedups over unoptimized MLIR code (baseline) achieved by our system (MLIR RL) in comparison to PyTorch and PyTorch compiler. 

|**NN Model**|**MLIR RL**|**PyTorch**|**PyTorch compiler**|
|---|---|---|---|
|ResNet-18|25.43|374.77|**411.26**|
|MobileNetV2<br>VGG|6.93<br>54.64|23.66<br>321.99|**28.23**<br>**328.77**|



TABLE III: Evaluating MLIR RL on different models. 

For all three models, the highest speedup is achieved by PyTorch and PyTorch compiler. For ResNet-18, PyTorch compiler is 16 _._ 17 _×_ faster than MLIR RL. For MobileNetV2, PyTorch compiler outperforms MLIR RL by 4 _._ 07 _×_ . For VGG, a 6 _._ 02 _×_ speedup is reached by PyTorch compiler. 

These differences arise primarily from PyTorch’s ability to achieve near-optimal performance in compute-intensive kernels 

||**Benchmark**|**MLIR RL**|**Mullapudi**|
|---|---|---|---|
||hexaquark-hexaquark (S = 12)|**13.25**|1.17|
||dibaryon-dibaryon (S = 24)|**7.57**|5.15|
||dibaryon-hexaquark (S = 32)|2.15|**4.68**|



TABLE IV: Comparison of speedups over MLIR baseline on LQCD benchmarks between MLIR RL (ours) and Halide autoscheduler (Mullapudi). _S_ refers to the input size. 

( _Matmul_ and _Conv2D_ ) which are the biggest bottleneck in these models. Additionally, our system still faces limitations vectorizing _Conv2D_ operations in these full models as it’s not yet capable of converting them to _Img2col + GEMM_ which greatly limits its potential to achieve higher speedups. 

_3) Evaluation on the LQCD Applications:_ We compare the performance of MLIR RL with the Halide autoscheduler [16] on the benchmark of LQCD applications. Table IV reports the speedups over unoptimized MLIR code (baseline). 

The table shows that MLIR RL achieves competitive performance across the LQCD applications. In particular, for the _hexaquark-hexaquark_ and _dibaryon-dibaryon_ applications, MLIR RL outperforms Halide’s autoscheduler by up to 11 _×_ . These gains stem from MLIR RL’s ability to apply loop tiling to reduce memory traffic in the deep nests of the LQCD correlator codes. It also selects loop orders that expose unit-stride vectorization and profitable outer-loop parallelism. 

## _D. Ablation Study_ 

In this section, we present an ablation study composed of three experiments: 1) we compare our two proposed interchange methods _Enumerated Candidates_ and _Level Pointers_ (presented in Sec IV-A1); 2) we compare the use of a multi-discrete action space compared to a flat action space; 3) we compare the use of immediate reward compared to final reward. 

For the first experiment, we train two agents, each with one of the proposed methods for loop interchange, and evaluate them on our benchmark suite. We observe that the agent trained with _Level Pointers_ achieves a higher average speedup of 18 _._ 7 _×_ over MLIR unoptimized code, compared to 14 _._ 5 _×_ for the _Enumerated Candidates_ . This highlights the efficiency of _Level Pointers_ in navigating the search space of interchanges. 

For our second experiment, we compared two identical RL agents but using different action space configurations. The first model uses a simple, flat action space, which presents a fixed set of transformation combinations, where a transformation and its parameters represent a single action (we call it Flat Action Space); the flat action space therefore has a high number of actions. The second model uses our proposed Multi-Discrete Action Space, where the agent first selects a transformation and then picks its parameters. 

Our results are presented in Figure 6. The figure illustrates that while the Multi-Discrete Action Space model converges more slowly, it is capable of exploring a wider and more diverse range of actions, leading to a higher average speedup at the end. This might be explained by the fact that learning in the first case is simpler, since the agent has fewer possible actions 

**==> picture [516 x 191] intentionally omitted <==**

Fig. 5: Speedups over MLIR baseline for each method across neural network operators, comparing our system (MLIR RL), Halide RL, PyTorch, and the PyTorch compiler. 

**==> picture [135 x 108] intentionally omitted <==**

Fig. 6: Comparison between the speedups achieved by training with a Flat Action Space and a Multi-Discrete Action Space. 

**==> picture [253 x 108] intentionally omitted <==**

Fig. 7: Comparison of ”Immediate Reward” and ”Final Reward” methods: the right plot shows the achieved speedup over training iterations; the left plot shows the achieved speedup over training time in hours. 

in each step. The results suggest that the Multi-Discrete Action Space provides a more robust framework for discovering better optimizations, albeit at the cost of longer training times. 

For the third experiment, we train two agents: one with an _Immediate Reward_ and the other with a _Final Reward_ , and evaluate them on the benchmarks. The results, shown in Figure 7, indicate that both reward functions achieve comparable performance in terms of average speedup. However, given the substantial execution overhead introduced by the _Immediate Reward_ approach (as discussed in Section IV-C), we prefer to use _Final Rewards_ . 

## VIII. CHALLENGES AND FUTURE WORK 

Developing MLIR RL presented two main challenges. First, designing an effective action space was time-consuming and required substantial domain expertise, amounting to roughly four person-years of effort. Second, training the RL agent is computationally expensive, and the need to retrain the agent many times during development made this a major bottleneck. Future work that reduces the cost of action-space design or accelerates training would be highly beneficial. 

## IX. CONCLUSION 

In this paper, we present MLIR RL, a Reinforcement Learning environment for automatic code optimization in the MLIR compiler. We propose a multi-discrete formulation of the action space where the action space is the Cartesian product of simpler action subspaces. We also propose a new method, called level pointers, to reduce the size of the action space related to the loop interchange transformation. These two methods enable more efficient and effective learning of the policy. Experimental results demonstrate that our proposed environment enables effective optimization of MLIR code, opening the door for more research on using RL for automatic code optimization. 

## X. DATA AVAILABILITY 

The official code for MLIR RL is released in the following repository: https://github.com/Modern-Compilers-Lab/ MLIR-RL. The artifact evaluation archive is available in: doi.org/10.5281/zenodo.17660414. 

## XI. ACKNOWLEDGMENT 

This research has been partly supported by the Center for Artificial Intelligence and Robotics (CAIR) at New York University Abu Dhabi, funded by Tamkeen under the NYUAD Research Institute Award CG010. The research was carried out on the High-Performance Computing resources at New York University Abu Dhabi. 

## ARTIFACT APPENDIX 

## _A. Abstract_ 

This artifact contains _MLIR RL_ , a deep reinforcement learning (RL) system that optimizes loop nests in MLIR. It includes the RL environment, pre-trained models, MLIR benchmarks, and scripts required to reproduce all results for MLIR RL (Figure 5 and Tables III–IV) and the PyTorch / PyTorch compiler results in Figure 5 and Table III. 

The artifact targets loop nests in the MLIR linalg dialect and uses a PPO-based actor–critic agent with a multi-discrete action space. It requires an LLVM/MLIR 19.x build (with Python bindings) and a Python 3.11 environment managed by Poetry. We provide a Docker image so experiments can be run on any Linux machine with Docker; the original experiments were run on an exclusive HPC node (Intel Xeon E5-2680 v4 @ 2.40GHz, 28 cores, 64 GB RAM). Preparing the environment takes about 1 hour and reproducing all evaluation results takes less than 1 hour. 

- _B. Artifact check-list (meta-information)_ 

- **Compilation:** LLVM/MLIR 19.x, built from source (Release, MLIR+Python bindings, X86 target); Clang/LLD; CMake+Ninja. 

- **Transformations:** Loop tiling, tiled parallelization, tiled fusion, interchange, vectorization. 

- **Binary:** Custom C++ MLIR tools (AstDumper, PreVec); standard MLIR tools from LLVM. 

- **Model:** Pre-trained PPO policy network (loaded via Python). 

- **Data set:** MLIR code files ( _≈_ 700 MB) + JSON baseline timings; downloaded from https://nyu.box.com/shared/static/ 5y1ilrccu3443dhcr854dt23uv0fysym.zip. 

- **Run-time environment:** Linux; Python 3.11 + Poetry; optionally Docker (recommended). 

- **Hardware:** CPU-only; evaluated on Intel Xeon E5-2680 v4 @ 2.40GHz, 28 cores, 64 GB RAM, exclusive node; similar multi-core x86-64 recommended. 

- **Run-time state:** Exclusive node, no competing jobs; stable CPU frequency for reliable timings. 

- **Execution:** Command-line scripts; _≈_ 1 hour for all evaluation runs. 

- **Metrics:** Execution time; speedup over MLIR baseline. 

- **Output:** JSON files with per-benchmark speedups for MLIR RL, PyTorch, and PyTorch compiler; Regenerated Figure 5 and Table III. 

- **Experiments:** Follow README, shell scripts in scripts/, config files in config/; no manual editing required. 

- **How much time is needed to prepare workflow (approximately)?:** _≈_ 1 hour. 

- **How much time is needed to complete experiments (approximately)?:** _≈_ 1 hour. 

- **Publicly available?:** Yes (Git repository + dataset link). 

- **Code licenses (if publicly available)?:** MIT Licence. 

- **Workflow framework used?:** Shell scripts + config files; optional Docker. 

- **Archived (provide DOI)?:** doi.org/10.5281/zenodo.17660414. 

## _C. Description_ 

- _1) How delivered:_ The artifact is delivered as a public Git 

- repository containing: 

- MLIR RL source code (Python + C++ tools) and configuration files under config/. 

- A Dockerfile to build a self-contained image. 

- Shell scripts in scripts/ for training, evaluation, and comparison. 

- Instructions and a link to the benchmark archive (MLIR files + JSON baseline timings). 

Users clone the repository, optionally build and run the Docker image, download the dataset into data/, and follow the README. 

- _2) Hardware dependencies:_ 

- Target: x86-64 CPU. 

- Tested: Intel Xeon E5-2680 v4 @ 2.40GHz, 28 cores, 64 GB RAM, exclusive node. 

- Recommended: _≥_ 16 physical cores and _≥_ 64 GB RAM for stable timings. 

- No GPU or accelerator required. 

- _3) Software dependencies:_ 

- Linux OS. 

- Either: 

   - Docker (recommended), using the provided Dockerfile, or 

   - Conda/Miniconda with Python 3.11, CMake, Ninja, Clang/LLVM, LLD, Poetry (versions as in README). 

- LLVM/MLIR 19.x built from source with MLIR and Python bindings enabled. 

- Python dependencies installed via poetry install. 

- _4) Data sets:_ 

- MLIR benchmarks (DL operators, DL models, LQCD kernels) as .mlir files. 

- JSON files with baseline execution times and benchmark lists for training/evaluation. 

- Distributed as a single archive ( _≈_ 700 MB) to be placed and unpacked in data/. 

## _D. Installation_ 

We recommend the Docker-based installation; the README gives full commands. 

_a) Docker (recommended).:_ 

- 1) Clone the repository and cd into it. 

- 2) docker build -t mlir-rl-artifact . 

- 3) docker run -it mlir-rl-artifact 

- 4) Inside the container, download and unzip the benchmark archive into data/. 

- _b) Without Docker (sketch).:_ 

- 1) Create and activate a Conda environment with Python 

- 3.11 and install the listed development packages. 

- 2) Clone and build LLVM/MLIR (release/19.x) with MLIR+Python bindings. 

- 3) Set environment variables (PATH, PYTHONPATH, LLVM_BUILD_PATH, MLIR_SHARED_LIBS) as in the README. 

- 4) Build custom tools in tools/ and create a .env file with their paths. 

- 5) Run poetry install in the repository root. 

- 6) Download and unzip the benchmark archive into data/. 

## _E. Experiment workflow_ 

All experiments are script-driven: 

- **Training (optional)** : ./scripts/train.sh Trains the RL agent from scratch (long, not required for AE). 

- **Evaluation (MLIR RL)** : ./scripts/evaluate.sh Evaluates a pre-trained model on the evaluation benchmarks and writes logs under results/. 

- **Paper results** : ./scripts/paper.sh 

   - Runs MLIR codes with no transformations to update the pre-recorded MLIR baseline. 

   - Runs MLIR RL, PyTorch, and the PyTorch compiler on the same benchmarks mentioned in the paper. 

   - Outputs speedups to JSON files in paper/results/ (mlir_rl.json for all MLIR RL results, torch_eager.json for PyTorch, and torch_jit.json for PyTorch Compiler); and Figure 5 and Table III in paper/figures/. 

- To reproduce the paper, evaluators: 

- 1) Run paper.sh. 

- 2) Observe the speedups in paper/results/ and figures in paper/figures/. 

- _F. Evaluation and expected result_ 

The artifact aims to reproduce: 

- **MLIR RL vs PyTorch/PyTorch compiler on operators** (Figure 5): Average speedups over the MLIR baseline for key operators (e.g., Matmul, Conv2D, Pooling). The values for Halide RL are included directly (in order to recreate the same figure), as it’s very complex and slow to reproduce results for Halide RL as well. 

- **MLIR RL vs PyTorch/PyTorch compiler on models** (Table III): Speedups for ResNet-18, MobileNetV2, and VGG. 

- **MLIR RL on LQCD applications** (Table IV): Speedups for Dibaryon–Dibaryon, Dibaryon–Hexaquark, and Hexaquark–Hexaquark. The table isn’t recreated here as the artifact doesn’t reproduce Halide Autoscheduler results, due to the complexity of its setup. 

Due to timing noise, we expect per-kernel execution times and speedups to vary within about _±_ 5% of the published numbers on similar hardware. For PyTorch and the PyTorch compiler, slightly larger deviations may occur: their performance depends on the exact PyTorch version, backend (oneDNN/MKL/OpenBLAS), and threading runtime, which can change kernel selection and fusion decisions. 

## _G. Experiment customization_ 

The system is configurable via JSON files in config/. Users can: 

- Change model and PPO hyperparameters (iterations, learning rate, batch sizes). 

- Modify the action space (e.g., allowed transformations, maximum number of loops). 

- Select different MLIR benchmarks. 

- Adjust environment variables (e.g., OMP_NUM_THREADS) and hardware settings. 

These knobs enable ablation studies and evaluation on additional benchmarks beyond those in the paper. 

## _H. Notes_ 

- For AE, we recommend using the provided pre-trained models and focusing on reproducing evaluation results rather than retraining. 

- Exclusive access to the node and stable CPU frequency significantly improve timing stability. 

- The top-level README provides concrete command examples and troubleshooting tips. 

## REFERENCES 

- [1] U. Bondhugula, A. Hartono, J. Ramanujam, and P. Sadayappan, “A practical automatic polyhedral parallelizer and locality optimizer,” in _Proceedings of the 29th ACM SIGPLAN Conference on Programming Language Design and Implementation_ , 2008, pp. 101–113. [Online]. Available: https://doi.org/10.1145/1375581.1375595 

- [2] S. Verdoolaege, J. Carlos Juega, A. Cohen, J. Ignacio Gomez,´ C. Tenllado, and F. Catthoor, “Polyhedral parallel code generation for cuda,” _ACM Trans. Archit. Code Optim._ , vol. 9, no. 4, jan 2013. [Online]. Available: https://doi.org/10.1145/2400682.2400713 

- [3] A. Adams, K. Ma, L. Anderson, R. Baghdadi, T.-M. Li, M. Gharbi, B. Steiner, S. Johnson, K. Fatahalian, F. Durand, and J. Ragan-Kelley, “Learning to optimize halide with tree search and random programs,” _ACM Trans. Graph._ , vol. 38, no. 4, jul 2019. [Online]. Available: https://doi.org/10.1145/3306346.3322967 

- [4] L. Zheng, C. Jia, M. Sun, Z. Wu, C. H. Yu, A. Haj-Ali, Y. Wang, J. Yang, D. Zhuo, K. Sen _et al._ , “Ansor: Generating high-performance tensor programs for deep learning,” in _14th USENIX symposium on operating systems design and implementation (OSDI 20)_ , 2020, pp. 863–879. [Online]. Available: https://doi.org/10.48550/arXiv.2006.06762 

- [5] R. Baghdadi, M. Merouani, M.-H. Leghettas, K. Abdous, T. Arbaoui, K. Benatchba, and S. Amarasinghe, “A deep learning based cost model for automatic code optimization,” in _Proceedings of the Fourth Conference on Machine Learning and Systems_ , vol. 3, 2021. [Online]. Available: https://doi.org/10.48550/arXiv.2104.04955 

- [6] M. Pecenin, A. M. Maidl, and D. Weingaertner, “Optimization of halide image processing schedules with reinforcement learning,” in _Anais do XX Simposio´ em Sistemas Computacionais de Alto Desempenho_ . SBC, 2019, pp. 37–48. 

- [7] B. H. Ahn, P. Pilligundla, A. Yazdanbakhsh, and H. Esmaeilzadeh, “Chameleon: Adaptive code optimization for expedited deep neural network compilation,” _CoRR_ , vol. abs/2001.08743, 2020. [Online]. Available: https://doi.org/10.48550/arXiv.2001.08743 

- [8] D. R. Lamouri, I. N. Aouadj, S. Kourta, and R. Baghdadi, “Pearl: Automatic code optimization using deep reinforcement learning,” in _Proceedings of the 39th ACM International Conference on Supercomputing_ , ser. ICS ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 959–974. [Online]. Available: https://doi.org/10.1145/3721145.3725766 

- [9] H. Shahzad, A. Sanaullah, S. Arora, R. Munafo, X. Yao, U. Drepper, and M. Herbordt, “Reinforcement learning strategies for compiler optimization in high level synthesis,” in _2022 IEEE/ACM Eighth Workshop on the LLVM Compiler Infrastructure in HPC (LLVMHPC)_ , 2022, pp. 13–22. [Online]. Available: https://doi.org/10.1109/ LLVM-HPC56686.2022.00007 

- [10] Q. Huang, A. Haj-Ali, W. Moses, J. Xiang, I. Stoica, K. Asanovic, and J. Wawrzynek, “Autophase: Compiler phase-ordering for hls with deep reinforcement learning,” in _2019 IEEE 27th Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM)_ , 2019, pp. 308–308. 

- [11] W. Huanting, T. Zhanyong, Z. Cheng, Z. Jiaqi, C. Chris, L. Hugh, and W. Zheng, “Automating reinforcement learning architecture design for code optimization,” in _Proceedings of the 31st ACM SIGPLAN International Conference on Compiler Construction_ , ser. CC 2022. New York, NY, USA: Association for Computing Machinery, 2022, p. 129–143. [Online]. Available: https://doi.org/10.1145/3497776.3517769 

- [12] C. Lattner, M. Amini, U. Bondhugula, A. Cohen, A. Davis, J. Pienaar, R. Riddle, T. Shpeisman, N. Vasilache, and O. Zinenko, “MLIR: Scaling compiler infrastructure for domain specific computation,” in _2021 IEEE/ACM International Symposium on Code Generation and Optimization (CGO)_ , 2021, pp. 2–14. [Online]. Available: https://doi.org/10.1109/CGO51591.2021.9370308 

- [13] C. Cummins, B. Wasti, J. Guo, B. Cui, J. Ansel, S. Gomez, S. Jain, J. Liu, O. Teytaud, B. Steiner _et al._ , “Compilergym: robust, performant compiler optimization environments for ai research,” in _2022 IEEE/ACM International Symposium on Code Generation and Optimization (CGO)_ . IEEE, 2022, pp. 92–105. [Online]. Available: https://doi.org/10.1109/CGO53902.2022.9741258 

- [14] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala, “Pytorch: An imperative style, highperformance deep learning library,” in _Advances in Neural Information Processing Systems 32_ . Curran Associates, Inc., 2019, pp. 8024–8035. [Online]. Available: https://doi.org/10.48550/arXiv.1912.01703 

- [15] J. Ansel, E. Yang, H. He, N. Gimelshein, A. Jain, M. Voznesensky, B. Bao, P. Bell, D. Berard, E. Burovski, G. Chauhan, A. Chourdia, W. Constable, A. Desmaison, Z. DeVito, E. Ellison, W. Feng, J. Gong, M. Gschwind, B. Hirsh, S. Huang, K. Kalambarkar, L. Kirsch, M. Lazos, M. Lezcano, Y. Liang, J. Liang, Y. Lu, C. K. Luk, B. Maher, Y. Pan, C. Puhrsch, M. Reso, M. Saroufim, M. Y. Siraichi, H. Suk, S. Zhang, M. Suo, P. Tillet, X. Zhao, E. Wang, K. Zhou, R. Zou, X. Wang, A. Mathews, W. Wen, G. Chanan, P. Wu, and S. Chintala, “Pytorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation,” in _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2_ , ser. ASPLOS ’24. New York, NY, USA: Association for Computing Machinery, 2024, p. 929–947. [Online]. Available: https://doi.org/10.1145/3620665.3640366 

- [16] R. T. Mullapudi, A. Adams, D. Sharlet, J. Ragan-Kelley, and K. Fatahalian, “Automatically scheduling halide image processing pipelines,” _ACM Transactions on Graphics (TOG)_ , vol. 35, no. 4, pp. 1–11, 2016. [Online]. Available: https://doi.org/10.1145/2897824.2925952 

- [17] F. Irigoin and R. Triolet, “Supernode partitioning,” in _(POPL’88)_ , San Diego, CA, Jan. 1988, pp. 319–328. [Online]. Available: https://doi.org/10.1145/73560.73588 

- [18] P. Feautrier, “Array expansion,” in _Proceedings of the 2nd international conference on Supercomputing_ . St. Malo, France: ACM, 1988, pp. 429–441. [Online]. Available: https://doi.org/10.1145/55364.55406 

- [19] M. E. Wolf and M. S. Lam, “A loop transformation theory and an algorithm to maximize parallelism,” _IEEE transactions on parallel and distributed systems_ , vol. 2, no. 4, pp. 452–471, 1991. 

- [20] V. Lefebvre and P. Feautrier, “Automatic storage management for parallel programs,” _Parallel Computing_ , vol. 24, pp. 649–671, 1998. [Online]. Available: https://doi.org/10.1016/S0167-8191(98)00029-5 

- [21] F. Quillere´ and S. Rajopadhye, “Optimizing memory usage in the polyhedral model,” _ACM Trans. on Programming Languages and Systems_ , vol. 22, no. 5, pp. 773–815, Sep. 2000. [Online]. Available: https://doi.org/10.1145/365151.365152 

- [22] W. Thies, F. Vivien, J. Sheldon, and S. Amarasinghe, “A unified framework for schedule and storage optimization,” in _Proc. of the 2001 PLDI Conf._ , 2001. [Online]. Available: https: //doi.org/10.1145/378795.378852 

- [23] A. Darte and G. Huard, “New complexity results on array contraction and related problems,” _J. VLSI Signal Process. Syst._ , vol. 40, no. 1, pp. 35–55, May 2005. [Online]. Available: http://dx.doi.org/10.1007/s11265-005-4937-3 

- [24] R. Baghdadi, “Improving tiling, reducing compilation time, and extending the scope of polyhedral compilation,” Ph.D. dissertation, Paris 6, 2015. 

- [25] R. Baghdadi, J. Ray, M. B. Romdhane, E. Del Sozzo, A. Akkas, Y. Zhang, P. Suriana, S. Kamil, and S. Amarasinghe, “Tiramisu: A polyhedral compiler for expressing fast and portable code,” in _2019 IEEE/ACM International Symposium on Code Generation and Optimization (CGO)_ . IEEE, 2019, pp. 193–205. [Online]. Available: https://doi.org/10.48550/arXiv.1804.10694 

- [26] R. Baghdadi, J. Ray, M. B. Romdhane, E. Del Sozzo, P. Suriana, S. Kamil, and S. P. Amarasinghe, “Tiramisu: A code optimization framework for high performance systems,” _arXiv preprint arXiv:1804.10694_ , 2018. [Online]. Available: https://doi.org/10.48550/arXiv.1804.10694 

- [27] K. Trifunovic, A. Cohen, D. Edelsohn, F. Li, T. Grosser, H. Jagasia, R. Ladelsky, S. Pop, J. Sjodin, and R. Upadrasta, “GRAPHITE two years after: First lessons learned from Real-World polyhedral compilation,” Jan. 2010. 

- [28] T. Grosser, A. Groslinger, and C. Lengauer, “Polly - performing polyhedral optimizations on a low-level intermediate representation.” _Parallel Processing Letters_ , vol. 22, no. 4, 2012. [Online]. Available: https://doi.org/10.1142/S0129626412500107 

- [29] T. Grosser, A. Cohen, J. Holewinski, P. Sadayappan, and S. Verdoolaege, “Hybrid hexagonal/classical tiling for gpus,” in _Proceedings of Annual IEEE/ACM International Symposium on Code Generation and Optimization_ , ser. CGO ’14. New York, NY, USA: ACM, 2014, pp. 66:66–66:75. [Online]. Available: https://doi.org/10.1145/2581122. 2544160 

- [30] N. Vasilache, O. Zinenko, T. Theodoridis, P. Goyal, Z. DeVito, W. S. Moses, S. Verdoolaege, A. Adams, and A. Cohen, “Tensor comprehensions: Framework-agnostic high-performance machine learning abstractions,” _CoRR_ , vol. abs/1802.04730, 2018. [Online]. Available: https://doi.org/10.48550/arXiv.1802.04730 

- [31] R. Baghdadi, A. Cohen, C. Bastoul, L.-N. Pouchet, and L. Rauchwerger, “The potential of synergistic static, dynamic and speculative loop nest optimizations for automatic parallelization,” 2011. [Online]. Available: https://doi.org/10.48550/arXiv.1111.6756 

- [32] L.-N. Pouchet, U. Bondhugula, C. Bastoul, A. Cohen, J. Ramanujam, P. Sadayappan, and N. Vasilache, “Loop transformations: Convexity, pruning and optimization,” in _38th ACM SIGACT-SIGPLAN Symposium on Principles of Programming Languages (POPL’11)_ . Austin, TX: ACM Press, Jan. 2011, pp. 549–562. [Online]. Available: https://doi.org/10.1145/1926385.1926449 

- [33] R. Baghdadi, A. N. Debbagh, K. Abdous, F. Z. Benhamida, A. Renda, J. E. Frankle, M. Carbin, and S. Amarasinghe, “Tiramisu: A polyhedral compiler for dense and sparse deep learning,” 2020. [Online]. Available: https://doi.org/10.48550/arXiv.2005.04091 

- [34] M. Merouani, M.-H. Leghettas, R. Baghdadi, T. Arbaoui, and K. Benatchba, “A deep learning based cost model for automatic code optimization in tiramisu,” Ph.D. dissertation, PhD thesis, 10 2020, 2020. 

- [35] T. Chen, L. Zheng, E. Yan, Z. Jiang, T. Moreau, L. Ceze, C. Guestrin, and A. Krishnamurthy, “Learning to optimize tensor programs,” in _Advances in Neural Information Processing Systems_ , 2018, pp. 3389–3400. [Online]. Available: https://doi.org/10.48550/arXiv.1805.08166 

- [36] C. Mendis, A. Renda, S. Amarasinghe, and M. Carbin, “Ithemal: Accurate, portable and fast basic block throughput estimation using deep neural networks,” in _International Conference on machine learning_ . PMLR, 2019, pp. 4505–4515. [Online]. Available: https://doi.org/10.48550/arXiv.1808.07412 

- [37] A. Brauckmann, A. Goens, and J. Castrillon, “Polygym: Polyhedral optimizations as an environment for reinforcement learning,” in _2021 30th International Conference on Parallel Architectures and Compilation Techniques (PACT)_ , 2021, pp. 17–29. [Online]. Available: https://doi.org/10.1109/PACT52795.2021.00009 

- [38] M. Merouani, K. A. Boudaoud, I. N. Aouadj, N. Tchoulak, I. K. Bernou, H. Benyamina, F. B.-S. Tayeb, K. Benatchba, H. Leather, and R. Baghdadi, “Looper: A learned automatic code optimizer for polyhedral compilers,” _arXiv preprint arXiv:2403.11522_ , 2024. [Online]. Available: https://doi.org/10.48550/arXiv.2403.11522 

- [39] Y. Hakimi, R. Baghdadi, and Y. Challal, “A hybrid machine learning model for code optimization,” _International Journal of Parallel Programming_ , vol. 51, no. 6, pp. 309–331, 2023. [Online]. Available: https://doi.org/10.1007/s10766-023-00758-5 

- [40] L. Mezdour, K. Kadem, M. Merouani, A. S. Haichour, S. Amarasinghe, and R. Baghdadi, “A deep learning model for loop interchange,” in _Proceedings of the 32nd ACM SIGPLAN International Conference on Compiler Construction_ , 2023, pp. 50–60. [Online]. Available: https://doi.org/10.1145/3578360.3580257 

- [41] N. Bendib, I. N. Aouadj, and R. Baghdadi, “A reinforcement learning environment for automatic code optimization in the mlir compiler,” _arXiv preprint arXiv:2409.11068_ , 2024. [Online]. Available: https://doi.org/10.48550/arXiv.2409.11068 

- [42] A. H. Ashouri, M. Elhoushi, Y. Hua, X. Wang, M. A. Manzoor, B. Chan, and Y. Gao, “Work-in-progress: Mlgoperf: An ml guided inliner to optimize performance,” in _2022 International Conference on Compilers, Architecture, and Synthesis for Embedded Systems (CASES)_ , 2022, pp. 3– 4. [Online]. Available: https://doi.org/10.1109/CASES55004.2022.00008 

- [43] Y. Zhao, H. Sharif, V. Adve, and S. Misailovic, “Felix: Optimizing tensor programs with gradient descent,” in _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3_ , ser. ASPLOS ’24. New York, NY, USA: Association for Computing Machinery, 2024, p. 367–381. [Online]. Available: https://doi.org/10.1145/3620666.3651348 

- [44] M. Merouani, A. Boudaoud, and R. Baghdadi, “Looperset: A large-scale dataset for data-driven polyhedral compiler optimization,” _arXiv preprint arXiv:2510.10209_ , 2025. [Online]. Available: https: //doi.org/10.48550/arXiv.2510.10209 

- [45] E. Park, L.-N. Pouchet, J. Cavazos, A. Cohen, and P. Sadayappan, “Predictive modeling in a polyhedral optimization space,” in _International Symposium on Code Generation and Optimization (CGO 2011)_ , 2011, pp. 119–129. [Online]. Available: https://doi.org/10.1109/CGO.2011.5764680 

- [46] T. Chen, T. Moreau, Z. Jiang, H. Shen, E. Q. Yan, L. Wang, Y. Hu, L. Ceze, C. Guestrin, and A. Krishnamurthy, “TVM: end-to-end optimization stack for deep learning,” _CoRR_ , vol. abs/1802.04799, 2018. [Online]. Available: https://doi.org/10.48550/arXiv.1802.04799 

- [47] A. Paliwal, F. Gimeno, V. Nair, Y. Li, M. Lubin, P. Kohli, and O. Vinyals, “Reinforced genetic algorithm learning for optimizing computation graphs,” 2020. [Online]. Available: https://doi.org/10.48550/arXiv.1905.02494 

- [48] G. He, S. Parker, and E. Yoneki, “X-rlflow: Graph reinforcement learning for neural network subgraphs transformation,” 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2304.14698 

- [49] The IREE Authors, “IREE,” Sep. 2019. [Online]. Available: https://github.com/iree-org/iree 

- [50] T. Jin, G.-T. Bercea, T. D. Le, T. Chen, G. Su, H. Imai, Y. Negishi, A. Leu, K. O’Brien, K. Kawachiya _et al._ , “Compiling onnx neural network models using mlir,” _arXiv preprint arXiv:2008.08272_ , 2020. [Online]. Available: https://doi.org/10.48550/arXiv.2008.08272 

- [51] The OpenXLA Authors, “Xla.” [Online]. Available: https://github.com/ openxla/xla 

- [52] O. Vinyals, M. Fortunato, and N. Jaitly, “Pointer networks,” in _Advances in Neural Information Processing Systems_ , C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, Eds., vol. 28. Curran Associates, Inc., 2015. [Online]. Available: https://doi.org/10.48550/arXiv.1506.03134 

- [53] S. R. Dubey, S. K. Singh, and B. B. Chaudhuri, “A comprehensive survey and performance analysis of activation functions in deep learning,” _CoRR_ , vol. abs/2109.14545, 2021. [Online]. Available: https://doi.org/10.48550/arXiv.2109.14545 

- [54] LLVM, “Torch-MLIR.” [Online]. Available: https://github.com/llvm/ torch-mlir 

- [55] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” in _Proceedings of the 34th International Conference on Machine Learning (ICML)_ , vol. 70. PMLR, 2017, pp. 4651–4660. [Online]. Available: https://doi.org/10.48550/arXiv. 1707.06347 

## APPENDIX 

## _A. Action History Encoding_ 

For each transformation type, we maintain a separate onehot encoded matrix, all sharing the same first dimension _τ_ , the maximum sequence length. At time step _t ∈_ [0 _, τ_ ), if a transformation is not applied, the entire slice [ _t,_ : _,_ :] is set to zero; otherwise, the slice encodes the chosen parameters. 

- _Tiling:_ stored in a matrix of shape _τ × N × M_ , where _N_ is the number of loops and _M_ the possible tile sizes. The slice [ _t,_ : _,_ :] records the selected tile size _m ∈_ [0 _, M_ ) for each loop _n ∈_ [0 _, N_ ), or is all zeros if no tiling is applied. 

- _Interchange:_ stored in a matrix of shape _τ × N × N_ . The slice [ _t,_ : _,_ :] specifies which loop _n ∈_ [0 _, N_ ) is placed in each position _i ∈_ [0 _, N_ ), or is zero if interchange is not applied. 

- _Terminal actions (No Transformation, Vectorization):_ no history is recorded, since applying them ends the optimization of the current operation and moves the agent to the 

next one. In case of vectorization, it’s considered terminal since a vectorized Linalg operation gets completely replaced by vector operations thus disabling any further Linalg transformations. 

## _B. Level Pointers Detailed Explanation_ 

For level pointers, completing an interchange action requires multiple inferences of the policy network. At the first step (when the agent decides to perform interchange), it selects a loop from the _N_ available loops to be placed at position 0. At each subsequent time-step _i ∈_ [1 _, N_ ), the agent is forced via the action mask to continue the interchange until completion. At each of these steps, it selects one of the remaining loops (with already chosen loops masked out) to be placed at position _i_ . After _N_ steps, this process produces a full permutation of the _N_ loops (as illustrated in Figure 8), which becomes the parameter of the interchange action. At this point, the agent is free to select any other transformation. To help the agent distinguish between intermediate sub-steps, partially selected loops are iteratively added to the interchange action history A, giving the agent information about what has already been chosen and the current stage of the permutation. 

**==> picture [250 x 157] intentionally omitted <==**

**----- Start of picture text -----**<br>
Level 1 Level 2 Level 3 Level 4<br>3 - 1 - 4 - 2<br>**----- End of picture text -----**<br>


Fig. 8: Illustration of the inference of interchange permutation using level pointers. The figure shows the output of the interchange head during the different sub-steps of the interchange action for a loop nest of size 4. 

_C. More Details about the Deep Learning Models_ 

||**Model**|**Total** <br>**Ops**|conv2d|pool|matmul|generic|unknown|
|---|---|---|---|---|---|---|---|
||MobileNetV2<br>ResNet<br>VGG|524<br>510<br>65|35<br>53<br>13|1<br>2<br>6|1<br>1<br>3|448<br>438<br>19|39<br>16<br>24|



TABLE V: Operation composition of the benchmarked neural network models. 

We use three widely used neural network architectures for our evaluation. 

- **VGG (Visual Geometry Group):** VGG is a deep convolutional network known for its simplicity and uniform architecture. It primarily consists of stacked convolutional layers with small 3 _×_ 3 kernels followed by max-pooling and fully connected layers. Despite its relatively high parameter count, VGG serves as a strong baseline for image classification tasks and is useful for evaluating performance on dense and repetitive patterns. 

- **ResNet (Residual Network):** ResNet introduces residual connections that allow gradients to flow more effectively through very deep networks. This innovation enables the construction of extremely deep models without suffering from vanishing gradients. Its architecture contains residual blocks, each of which includes skip connections that bypass one or more layers. ResNet is widely used in both academic and industrial applications due to its robust performance and efficient training behavior. 

: **tensor** <8x8xf64> to vector<8x8xf64> %mul = arith.mulf %vA, %vB : vector<8x8x512xf64> %red = vector.multi_reduction <add>, %mul, %vC [2] : vector<8x8x512xf64> to vector<8x8xf64> %C2 = vector.transfer_write %red, %C[0, 0] : vector<8x8xf64> to **tensor** <8x8xf64> scf.forall.in_parallel { **tensor** .parallel_insert_slice %C2 into %acc[%i0, %j0] [8, 8] [1, 1] : **tensor** <8x8xf64> into **tensor** <256x1024xf64> } } %t2 = **call** @nanoTime() : () -> i64 %dt = arith.subi %t2, %0 : i64 **return** %1, %dt : **tensor** <256x1024xf64>, i64 } } 

- **MobileNet:** MobileNet is a lightweight neural network architecture optimized for mobile and embedded devices. It achieves efficiency by using depthwise separable convolutions, which reduce both the computational cost and the number of parameters. MobileNet is particularly suitable for performance benchmarking in low-resource environments, and it offers a useful contrast to the heavier VGG and ResNet models. 

The composition of each model in terms of operation types is detailed in table V. 

## _D. Linalg Operation After Optimization_ 

Listing 2: Optimized MLIR matmul after tiling and vectorization. 

#map = **affine_map** <(d0) -> (d0 * 8)> #map1 = **affine_map** <(d0, d1) -> (d0, 0, d1)> #map2 = **affine_map** <(d0, d1) -> (0, d1, d0)> **module** { func.func private @nanoTime() -> i64 attributes {llvm. emit_c_interface} func.func @main(%arg0: **tensor** <256x512xf64>, %arg1: **tensor** <512x1024xf64>, %arg2: **tensor** <256x1024xf64>) -> ( **tensor** <256x1024xf64>, i64) attributes {llvm.emit_c_interface} { %0 = **call** @nanoTime() : () -> i64 %1 = scf.forall (%i, %j) in (32, 128) shared_outs(%acc = %arg2) -> ( **tensor** <256x1024xf64>) { %i0 = affine.apply #map(%i) %j0 = affine.apply #map(%j) %A = **tensor** .extract_slice %arg0[%i0, 0] [8, 512] [1, 1] : **tensor** <256x512xf64> to **tensor** <8x512xf64> %B = **tensor** .extract_slice %arg1[0, %j0] [512, 8] [1, 1] : **tensor** <512x1024xf64> to **tensor** <512x8xf64> %C = **tensor** .extract_slice %acc[%i0, %j0] [8, 8] [1, 1] : **tensor** <256x1024xf64> to **tensor** <8x8xf64> %zero = arith.constant 0.0 : f64 %vA = vector.transfer_read %A[0, 0], %zero {permutation_map = #map1} : **tensor** <8x512xf64> to vector<8x8x512xf64> %vB = vector.transfer_read %B[0, 0], %zero {permutation_map = #map2} : **tensor** <512x8xf64> to vector<8x8x512xf64> %vC = vector.transfer_read %C[0, 0], %zero 

