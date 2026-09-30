# Astra Pro 数据处理层

更新日期：2026-09-21。任务三。新增`camera_capture/processing.py`，并修正`core/depth_pipeline.py`的原始深度路径；尚未接入实时检测页面，也没有完成空间配准。

## 数据约定

- `normalize_depth(image, unit_m, config)`显式接收每个原始数值对应多少米。支持二维uint16和float32/64；拒绝8位预览。统一输出float32米、有效像素掩码及有效率，无效距离为NaN。
- 零、负数、NaN、无穷、uint16最大值65535，以及范围外值均无效。默认范围0.6～6米是可配置工程工作范围，并非本轮测距标定结论。
- `depth_preview()`使用固定物理范围生成伪彩色，无效像素为黑色。预览不用于运动计算；单个极值改变不会重新缩放整幅图。
- 核心接口继续兼容uint16毫米输入。浮点深度必须通过`depth_unit_m`声明单位；不再按“是否大于255”猜测距离。浮点米输入传`depth_unit_m=1.0`。

## 软件时间配对

深度SDK微秒时间在前8个有效样本中估算固定偏移，映射到本机单调时钟。彩色使用接收完成的本机单调时间，减去可配置的`color_receive_offset_s`（默认0，尚未校准）。普通接收抖动在容差内不会触发预热；映射时间不会晚于实际接收时刻。后续时钟残差绝对值超限会清空待配对与运动状态，重新预热，并递增`clock_epoch`。

每路队列默认最多8帧；从当前缓存选择时间差最小的一对，每帧仅使用一次。默认容差50毫秒、过期上限250毫秒。重复或乱序帧、时间回退、会话过期、设备标识变化、格式不匹配以及过期数据均不产生有效配对。超容差的帧等待可匹配帧，仍受缓存上限和到期清理约束。

**这属于软件接收时钟估计，不等于曝光同步。** UVC当前没有可核验的曝光时间，接收与曝光之间还可能存在系统缓存和解码延迟。因此50毫秒约束的是估计时间差，不能据此承诺真实曝光偏差。结果明确包含`timing_basis`、`hardware_synchronized=False`、`exposure_alignment_verified=False`；也不宣称空间对齐。

`FramePreprocessor`供单一后端消费者使用。`poll(capture)`读取采集层的有限缓冲快照，并跳过已经观察过的帧；多页面后续应读取同一个后端处理结果。采集停止或故障时清空待配对状态；重新启动的session与时钟epoch必须被下游用于重置时序状态。

## 质量和运动

运动仅比较相邻已接受帧的共同有效像素，使用实际时间差计算米/秒及相对距离变化率。无效像素出现/消失不计为运动。均匀但有效的深度平面不会因为灰度对比度低而被判为无效。

默认检查：有效率至少8%；运动差值阈值0.05米；大量像素（默认60%共同有效像素）出现超过1.5米的突变时拒绝并重新预热；超过0.75秒的时序间隔不跨间隔计算运动。所有参数位于`config.yaml`的`multimodal`节，不是临床阈值。

每个配对携带有效率、异常跳变比例、真实帧间隔、时序是否就绪与原因。首帧、形状变化、大间隔及异常后的预热不产生有效运动证据；全无效数据返回不可用。突变拒绝是保守工程门控，可能同时拒绝真实快速变化，仍需后续场景评估与调参。

## 提供给算法的入口

```python
from camera_capture.processing import FramePreprocessor

processor = FramePreprocessor(config.multimodal)
pair = processor.poll(camera)
if pair is not None:
    # 若过期、质量不足或尚未完成时序预热，此方法抛出ValueError。
    inputs = pair.algorithm_inputs()
    # inputs含rgb_frame(BGR)、depth_frame(米)、timestamp_s和depth_unit_m=1.0。
    # 后续任务四将把它交给process_rgb_depth，并处理回退与页面展示。
```

原始输入、米制数组与预览分开。`algorithm_inputs()`再次检查时效与运动质量，不会把等待、丢帧或无效深度包装成安全零风险。它不自动调用模型、不写数据库、不更新个人基线。

## 独立实机检查

在`C:\safe`运行：

```powershell
.\.venv\Scripts\python.exe -B scripts\check_astra_data.py --sdk-dir 'E:\orb\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs' --seconds 30
```

可选`--output`只保存统计JSON。检查不加载检测模型、不保存画面、不访问正式数据库。停止后释放相机。

## 本阶段边界

保留原有8位组合视频的预览证据处理，原始深度使用独立的米制分析路径。RGB检测、融合权重、助手与RAG未修改。当前实时页面仍为RGB；软件配对通过不能证明曝光同步、三维人体测量、检测准确率或临床有效性。
