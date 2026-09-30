# Astra Pro 后端采集模块

适用范围：任务二采集层；任务三补充见本文末尾。更新日期：2026-09-21。目标应用：`C:\safe`。本阶段完成独立采集，不接入实时页面、检测算法、数据库或个人基线。

## 使用环境

- Windows 64位、现有项目Python环境。
- 已安装Orbbec 4.3.0.22驱动和用户提供的OpenNI 2.3.0.86 beta6 Win64 SDK。
- 使用现有numpy、imageio-ffmpeg依赖，不安装新模型或替换检测依赖。
- `--sdk-dir`指向包含`OpenNI2.dll`与`OpenNI2/Drivers`的目录，而非SDK顶层目录。

本机SDK路径：`E:\orb\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs`。

## 命令

在`C:\safe`中运行。检查前关闭其他占用Astra的查看器。

```powershell
.\.venv\Scripts\python.exe -B scripts\check_astra_capture.py --sdk-dir 'E:\orb\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs' --list
.\.venv\Scripts\python.exe -B scripts\check_astra_capture.py --sdk-dir 'E:\orb\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs' --seconds 15 --restart
```

第一条只枚举设备。第二条运行三轮：首次采集、停止后重新启动、断开后重新连接。退出时释放设备。仅打印统计；指定`--output`才将统计写入JSON，不保存图像或录像。不要同时运行两份检查。

## 后端接口

```python
from pathlib import Path
from camera_capture import AstraCapture, CaptureConfig

config = CaptureConfig(sdk_dir=Path(r'E:\orb\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs'))
with AstraCapture(config) as camera:
    selected = camera.connect()
    camera.start()
    depth = camera.latest('depth')
    color = camera.latest('color')
    status = camera.status()
    camera.stop()
```

- 生命周期：`disconnected → connected → starting → running`；停止返回`connected`，`close()`断开。异常进入`error`并使缓存失效；调用`stop()`/`close()`清理后再连接，不会自动重连。
- `connect()`使用OpenNI设备URI与UVC的DirectShow alternative ID，不用易变的摄像头序号。只选择深度VID/PID `2bc5:0403`和色彩`2bc5:0501`。没有匹配设备就报错，不回退到笔记本摄像头。
- 多台设备时拒绝自动猜配，需显式传入`DeviceSelection`。用户负责核对两路属于同一物理设备；本阶段未建立多设备USB父节点配对。
- 一个Windows用户下仅允许一个模块实例占用Astra，包括跨进程。保守锁在连接时获取，断开时释放，停止但未断开仍保留。不同页面应复用一个后端管理实例，并由后续页面层维护操作归属；本阶段没有页面代码。
- 锁防止本模块重复占用，不控制其他厂商查看器或浏览器。其他程序占用时采集会明确失败。异常终止整个进程时可能需关闭残留外部采集进程；常规`close()`与正常解释器退出会执行清理。
- 每路各一条持续读取线程，彩色由独立FFmpeg进程解码；检测与页面读取不会阻塞采集。每路默认最多保留2帧，满时覆盖最旧帧。缓冲覆盖数是消费者缓存更新统计，不代表相机丢帧数。
- `latest()`返回最新帧的独立副本；停止或过期返回`None`，采集故障抛异常，不能将其解释成零风险。通过`after_sequence`避免消费同一帧；重新启动产生新`session_id`，消费者应重置序号游标。

## 帧约定与边界

| 字段 | 含义 |
| --- | --- |
| `image` | 深度为二维`uint16`毫米数组；彩色为三通道`uint8` BGR |
| `sequence` | 深度采用SDK帧号，彩色为本次采集的接收计数 |
| `received_monotonic_s` | 本机读取完成的单调时钟；不是曝光时间 |
| `device_timestamp_us` | 深度SDK设备时间；彩色为`None` |
| `timestamp_basis` | 明确时间来源，不假造彩色硬件时间戳 |
| `depth_unit_m` | 深度为`0.001`；彩色为`None` |
| `device_id` / `session_id` | 设备来源与本次启动标识 |

本阶段固定为已验证的640×480、30 FPS；原始深度按SDK格式100强制毫米单位，零值原样保留为无效距离。没有执行两路同步、配准、深度质量评分、人体关联或地面标定。分别读取的两路最新帧**不是同步帧对**，`status()['synchronized']`固定为`False`。

持续超时会使缓存失效并结束采集循环。若极端情况下原生SDK线程无法退出，模块拒绝释放仍被访问的句柄与占用锁，并提示关闭采集进程，避免误认为设备已释放。

当前浏览器实时页仍走原有RGB流程。仅添加本模块不会让检测使用深度；后续还须完成数据时间配对、质量门控和实时检测接入。


## 任务三后续更新（2026-09-21）

在独立采集模块之上已增加软件时间配对、深度有效值与质量处理。采集接口本身仍返回未配对帧，使用新的`FramePreprocessor`获取软件配对。以上关于“未配对”的描述适用于采集层；数据层详见[Astra数据处理说明](Astra数据处理说明.md)。硬件曝光同步、空间配准和实时页面接入仍未完成。
