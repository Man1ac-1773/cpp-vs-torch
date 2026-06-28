# 03. End-to-End ML Training: The Reality of Overhead

After optimizing matrix multiplication, I built an entire ML framework from scratch to test these engines in a real training loop. 

I implemented a full Autograd engine, Rectified Linear Unit (ReLU), Mean Squared Error (MSE), and Stochastic Gradient Descent (SGD). I trained a 2-Layer Multi-Layer Perceptron (Batch Size: 1024) for 100 Epochs.

Here are the final training times across the engines (using their best multi-threaded SIMD backends on the `performance-plugged` profile):

| Framework / Engine | Total Training Time (100 Epochs) | Memory Management Strategy |
| :--- | :--- | :--- |
| **NumPy (OpenBLAS)** | **0.127s** | Pre-allocated Numpy Arrays |
| **My C++ Engine (`minigrad`)** | **0.371s** | `std::vector` Heap Allocations |
| **My C Engine (`macrograd`)** | **1.70s** | $O(1)$ Arena Checkpointing |
| **PyTorch (ATen C++ Backend)** | **1.94s** | Caching Allocator + Autograd Graph |

## The Truth About Memory vs. Framework Overhead

Initially, I hypothesized that my C Engine's $O(1)$ Arena Checkpoint (which avoids heap allocations by just resetting a pointer every epoch) would easily beat standard `std::vector` allocations in C++. 

The data proves that assumption entirely wrong.

The C++ engine (`0.37s`) vastly outperformed the C Arena Checkpoint engine (`1.70s`). Why? Because for a tiny 2-layer MLP, memory allocation is *not* the primary bottleneck. The overhead of threading paradigms (OpenMP vs Pthreads) and how well the compiler optimizes the surrounding loops completely eclipsed any minor gains from the Arena allocator.

### Why is PyTorch so slow here? And why is NumPy so fast?

You'll notice that my C++ engine vastly outperformed PyTorch, but it got absolutely destroyed by NumPy. 

At first glance, it seems absurd that PyTorch (the industry standard) is the slowest engine on the board. However, PyTorch is carrying massive framework overhead. For every forward pass, PyTorch dynamically constructs a complex C++ Autograd computation graph. It also suffers from Python dispatch overhead on every single tensor operation. For a massive ResNet on a GPU, this overhead is negligible. But for a tiny 100-epoch dummy MLP taking milliseconds, the framework overhead dominates the runtime.

NumPy, on the other hand, just executes a raw C loop backed by heavily optimized OpenBLAS math. Since we manually calculate the gradients in the NumPy script, there is no Autograd graph being constructed in the background. It is pure, raw, unadulterated math, which is why it runs in a blistering 0.127 seconds.

**Key Learning:** Don't optimize memory allocations before measuring if they are actually the bottleneck. And always remember that massive ML frameworks carry significant baseline overhead that makes them look artificially slow on micro-benchmarks.

---
### Scripts & Raw Data
- **C Engine Script**: [`c_train.cpp`](../benchmarking/training/c_train.cpp)
- **C++ Engine Script**: [`cpp_train.cpp`](../benchmarking/training/cpp_train.cpp)
- **PyTorch Script**: [`py_train.py`](../benchmarking/training/py_train.py)
- **NumPy Script**: [`np_train.py`](../benchmarking/training/np_train.py)
- **Raw Data Log**: [`benchmarking/training/data/`](../benchmarking/training/data/)
