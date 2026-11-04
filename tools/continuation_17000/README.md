# 时序一致的10/90增强

`augmentation.py`包含增强采样、执行、遮挡保护和训练图像入口的安装函数。增强参数按draw生成并传给当前/未来图像；策略、深度和视频教师共享同一增强RGB。

此目录公开算法模块和CPU测试。训练环境中需要PyTorch、NumPy、Pillow、OpenCV；`install()`需要已安装本仓库的LingBot数据管线。无需模型权重即可运行测试：

```bash
python tools/continuation_17000/test_augmentation.py
```

配方、实验启用阶段和限制见[技术设计](../../docs/design.md)。测试覆盖概率、形状与dtype、随机重放、当前/未来一致性、遮挡保护及补黑边界。
