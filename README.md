# yet another halo engine
Modern custom inference engine targeted at Strix Halo, targeting both NPU and GPU inference

Goals:
- Hybrid NPU+GPU inference, and a fast NPU path
- Heavily tuned GEMM kernels for ROCm HRX
- Improve prefill speeds by 1.5x over other tuned inference backends
- Minimal scope in models, quantization formats, etc. to keep development fast