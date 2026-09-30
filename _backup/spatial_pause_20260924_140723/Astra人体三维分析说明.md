# Astra Pro 人体三维分析

更新：2026-09-23。正式应用：`C:\safe`。

## 当前完成程度

已增加标定读取与校验、深度到彩色的三维投影和Z-buffer、单人躯干深度提取、地面平面估计、躯干离地高度、下降速度、低位持续时间与低位活动速度，以及页面和事件来源快照。

**当前为三维特征观察模式，不改变跌倒/前置风险融合权重，不参与个人基线。**现有场景级深度融合继续按原规则工作，不能把它解释为人体级三维融合。尚无本机经验证的标定文件和米制三维标注数据，因此当前现场的人体三维结果应显示不可用。这不是三维检测效果验收完成。

用户确认相机已固定、能看到地面，暂时没有已知格子尺寸的棋盘格。本次真实探测：沙箱外可枚举深度和 Astra Pro HD Camera 彩色设备；SDK property 14 返回参数含非有限值，拒绝当作有效标定。短时采集工具成功保存软件配对帧。不具备真实空间配准、地面高度真值或检测性能结论。

## 页面操作

1. 完成下述标定后，将经验证文件安装到 `data/calibration/astra_spatial.json`。不要把示例、模拟数据、出厂原始参数直接放到此路径。
2. 选择 Astra 来源，连接设备。只有核实相机安装位置和倾角与标定时一致，才勾选“已核对相机位置、倾角与地面标定时一致”。开始监测时加载文件。
3. 设备详情显示三维可用性和原因。合格单人显示躯干离地米数、下降速度m/s、低位持续秒数、低位活动m/s、有效覆盖率。下降速度正值表示向地面下降；初帧或时间间隔过长显示预热。
4. 多人时明确停用人体三维归属；没有人、关键点不足、深度空洞/混合、设备不匹配、失步或异常跳变均不输出有效高度。已有RGB/场景深度检测依原规则继续。
5. 换设备、分辨率、RGB裁切/镜像设置须重新空间标定；移动或倾斜相机须重新地面标定。程序不能自动证明固定支架没有移动。采集会话变化会取消三维标定使用，须停止、重新核对后开始。

人体3D字段随告警/前置风险等现有记录的 `capture_context.body_3d` 保存，包含标定SHA256、安装位置、配准误差、人物track_id、质量和特征；历史详情可查看。旧记录不补写3D信息。没有新记录触发时不会逐帧写数据库。CSV既有摘要列不扩展为完整3D时序导出。

## 标定工具

在 `C:\safe` 打开 PowerShell，先在页面断开相机。以下输出路径不能已存在，工具拒绝覆盖。

```powershell
.\.venv\Scripts\python.exe -B scripts\astra_spatial_calibrate.py capture --factory-only --output data\calibration\factory.json
.\.venv\Scripts\python.exe -B scripts\astra_spatial_calibrate.py capture --output data\calibration\capture_01.npz
```

`capture` 最长等候15秒获取质量合格的软件配对数据，保存 `color_bgr`、`depth_m`（NaN无效）、设备标识、捕获ID、两路时间和配对偏差，之后释放设备。数据仅在本机。每次命令使用新的输出名；静止目标可减小两路曝光不同步的影响。

出厂读取ABI依据已安装SDK的 `Win64-Release/sdk/Include/OniCTypes.h` 的 `OBCameraParams`（120字节）和 `OniCProperties.h` 的 `OBEXTENSION_ID_CAM_PARAMS=14`，与 `samples/samples/SimpleViewer/AstraProD2C.cpp` 的普通Astra读取路径相同。原始左右内参、畸变、r2l外参不自动解释为当前UVC的已验证参数。非有限值或ABI大小不符均拒绝。

### 1. 获得内参

优先向设备厂商取得该设备、该分辨率的有效标定。当前无有效出厂参数时，需要自行标定。可以打印棋盘并用尺实测方格边长；打印设置必须100%，实际测量比文件标称尺寸优先。需要平整板，RGB和同深度光路的IR图像分别标定，不能用伪彩色深度图代替IR图像。当前采集模块未开放IR拍照，需用SDK原生查看器采集IR标定图，或由厂商提供深度内参。

离线内参命令：

```powershell
.\.venv\Scripts\python.exe -B scripts\astra_spatial_calibrate.py intrinsics --input data\calibration\rgb_chessboard.json --output data\calibration\rgb_intrinsics.json
```

输入JSON：`{"inner_corners":[8,5],"square_size_m":0.02,"images":["rgb_01.png","rgb_02.png"]}`。图片相对于JSON目录；示例列表需扩充为至少12张不同位置、距离、倾角的成功棋盘图。重复视角不能替代覆盖；RMS只是拟合误差，仍须独立验证。深度IR内参同样格式另行生成。

### 2. 拟合外参并验证空间配准

提供静态目标的深度像素u/v/轴向z米，以及同一物理点的彩色u/v。必须来自可辨认的同一目标角点/特征，不用同尺寸坐标直接当作对应点。深度边缘空洞、遮挡及两光路不可共同见到的点不使用。用于拟合和验证的目标采集分开，覆盖画面和多个距离。

`extrinsics --input fit.json --output fitted.json` 用PnP RANSAC拟合深度到彩色的外参。输入字段：

```json
{
  "capture_id": "拟合采集ID",
  "device_ids": {"depth": "实际OpenNI URI", "color": "实际UVC ID"},
  "depth": {"size": [640,480], "matrix": [[580,0,320],[0,580,240],[0,0,1]], "distortion": [0,0,0,0,0]},
  "color": {"size": [640,480], "matrix": [[580,0,320],[0,580,240],[0,0,1]], "distortion": [0,0,0,0,0]},
  "fit_depth_uvz_m": [],
  "fit_color_uv": [],
  "verification": {"independent_capture": true, "capture_id": "独立验证采集ID", "depth_uvz_m": [], "color_uv": []}
}
```

**矩阵数字仅说明格式，不是这台相机的参数，严禁直接使用。**每个点数组至少12行；验证点覆盖图像宽和高各25%以上，并覆盖至少0.3m深度变化。软件计算独立对应点重投影P95≤3px、最大≤6px，否则拒绝。阈值在config中，为工程门槛，非实测精度承诺。拟合与验证不能使用同一采集ID。

最终坐标约定：`schema=1`、`convention="opencv_depth_to_color_m"`，x向右、y向下、z向前；`rotation` 为3×3正交矩阵，`translation_m` 为3元素米制向量。深度先去畸变反投影，再外参变换、RGB畸变投影，冲突像素只保留最近表面。不填补遮挡孔洞，不以resize完成配准。

### 3. 标定地面

保持相机固定，在空场景采集两份独立NPZ，一份拟合、一份验证。人工明确圈出真实地面在**原始深度图**上的多边形，排除床、桌、墙、人和障碍物。选区JSON：`{"depth_polygon":[[100,300],[540,300],[600,470],[40,470]]}`，示例坐标不能直接用于现场。

```powershell
.\.venv\Scripts\python.exe -B scripts\astra_spatial_calibrate.py floor --input data\calibration\fitted.json --floor-capture data\calibration\floor_fit.npz --floor-validation data\calibration\floor_check.npz --region data\calibration\floor_region.json --placement room_a_fixed_01 --confirm-floor --output data\calibration\astra_spatial.json
.\.venv\Scripts\python.exe -B scripts\astra_spatial_calibrate.py verify --input data\calibration\astra_spatial.json --output data\calibration\verification_report.json
```

工具将地面点变换至彩色相机坐标，RANSAC拟合、SVD细化；要求至少100点、80%内点、足够二维覆盖、相机离地0.3–3m、合理朝向，独立地面验证P95残差≤2cm。严格限制大俯仰安装，未通过应调整安装或开展单独验证，不能放宽门槛伪造通过。算法不能辨别操作员选的是不是地板，`--confirm-floor` 是真实现场确认。

完整文件保存floor.points_color_m、validation_points_color_m、operator_confirmed_floor、placement_id和捕获ID，加载时重新计算验证；不能只写 `verified:true`。

## 特征与限制

采用高置信度双肩双髋内缩后的躯干区域；至少12个有效点、有效覆盖≥50%、深度P90−P10≤0.15m。用三维中位点对地面有向距离作为“躯干离地高度”，不是头顶高度或人体最小离地高度。宽松衣物、遮挡和背景混入仍可能影响测量，当前无身体分割及多人深度归属。

每个有效track使用真实dt求高度下降速度、三维位移活动速度；默认躯干低于0.55m累计低位时间。超过0.75秒间隔、换人、失效和异常位置跳变清除短期历史。track_id是短期视觉跟踪号，不是确认身份，遮挡换人仍有跟踪局限。低位不等于跌倒。

## 标注与融合调参

已有URFD等RGB事件标注没有本机米制三维轨迹、标定和地面真值，不能据此调整本次三维权重。新增 `scripts/evaluate_spatial_features.py` 只离线评估候选规则，输出建议和验证指标，**不会写config或改变实时融合**。

输入CSV字段：`source,reviewed,calibration_id,split,subject_id,clip_id,label,torso_height_m,descent_speed_m_s,low_duration_s`。source只能astra_metric，reviewed为true，标定ID为SHA256，split为train/validation，label为fall_evidence/non_fall。人员和片段不能跨集合；不可用特征单独统计，不能填0。命令：

```powershell
.\.venv\Scripts\python.exe -B scripts\evaluate_spatial_features.py --input data\calibration\labelled_features.csv --output data\calibration\shadow_evaluation.json
```

候选低位阈值0.35/0.45/0.55m、速度0.3/0.6/0.9m/s、持续1/2/3s仅是离线探索网格。按训练集F1选取，再在独立验证集计算逐帧TP/FP/FN/TN；不是事件级检测准确率。真实融合前仍须补齐同步RGB/深度标注，做事件级RGB基线与三维融合对照、困难负例、不可用率及独立人员测试，然后人工审阅阈值和权重。不能要求老年人模拟跌倒。

## 验证边界

合成几何测试验证投影、遮挡、单位/证据拒绝、地面拟合和运动状态；fake会话验证降级与记录，不代表真实几何精度。当前实测仅验证设备枚举、出厂参数拒绝和帧对采集。现场空间配准、地面真值、多姿态人体测量和融合调参尚待有效标定及标注完成。
