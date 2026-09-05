# 外部专项验收数据

此目录仅用于本地验收，原始数据不进入版本库。每个数据子目录只保留来源清单；下载包、解压文件和派生结果均被 `.gitignore` 排除。

当前接入的 UTD Continuous Transition Movements 数据含 Kinect v2 原始深度和惯性序列。它用于验证 16 位/毫米级深度读取、质量判定、距离变化和前置融合链路；公开文件没有逐动作起止真值，因此不能据此计算近跌倒或 Sit-to-Stand 的 Precision、Recall、F1。

CAUCAFall 子目录本地保存 10 段真实人体“坐下”视频；其中 6 段由 OmniFall 统一标注提供 `stand_up` 逐事件起止时间。原视频遵循 CAUCAFall 页面标明的 CC BY 4.0，统一时间标注来自 OmniFall。可复核清单位于 `test_videos/real_pre_fall_clips.csv` 和 `test_videos/real_pre_fall_event_annotations.csv`。

UMAFall 官方 ADL 视频经筛选只有动作示例，没有本项目需要的逐事件时间标注和近跌倒正样本，因此没有纳入或保留在本轮定量验收集。

SisFall 子目录本地保存 4 段官方活动视频：D18 `Stumble while walking` 是真实人体模拟近跌倒正样本；D01 慢走、D02 快走和 D19 原地轻跳是同源负样本。SisFall 论文及补充数据采用 CC BY 4.0。D18 的活动类别来自官方定义，近跌倒事件起止时间由本项目按 `test_videos/real_pre_fall_annotation_protocol.md` 逐帧复核；这两层证据在清单中分别记录，不把人工时间边界误写成官方逐帧标注。

URFD 子目录保存官方 `urfall-cam0-falls.csv` 和 `urfall-cam0-adls.csv`。其中跌倒文件逐帧给出 `-1`（未躺地）、`0`（跌倒中）、`1`（已躺地）标签；本项目用首次 `0` 到首次 `1` 派生6段跌倒事件区间，并通过视频帧数逐段核验。专项视频清单、事件真值和标注规则分别位于 `test_videos/specialized_scene_clips.csv`、`test_videos/specialized_event_annotations.csv`、`test_videos/specialized_scene_annotation_protocol.md`。
