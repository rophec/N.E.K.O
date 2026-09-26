# NVIDIA CUDA 可选运行组件

Windows 正式包默认使用 `onnxruntime-directml==1.24.4`，兼容 AMD、Intel 与 NVIDIA GPU。
需要 NVIDIA CUDA 的构建可在 `Build Desktop App` 手动工作流中选择
`windows_onnxruntime=cuda`；构建过程会先移除所有冲突的 ONNX Runtime
发行包，再安装 ABI 对齐的 `onnxruntime-gpu==1.24.4`。

CUDA 版需要 NVIDIA 驱动以及与 ONNX Runtime 1.24.4 匹配的 CUDA 12.x、
cuDNN 9 运行库。运行时面板会显示实际加载的 provider；CUDA 不可用时会明确
显示 CPU 降级原因，不会把“极速模式”误报成 CUDA。
