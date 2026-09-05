暮安智护——测试视频使用说明
============================

一、推荐使用的文件夹

请优先使用：

  00_ready_to_upload\01_fall_positive
  00_ready_to_upload\02_daily_activity_negative

其中：

  POS = 跌倒正例，系统应当识别到跌倒风险或产生跌倒告警。
  NEG = 日常活动负例，系统不应产生红色跌倒告警。

可在系统“上传视频检测”页面中一次选择一段 MP4 视频进行测试。

二、目录说明

  00_ready_to_upload
    已从原始多模态画面中截取普通 RGB 彩色画面，并放大为
    640×480、30 FPS、H.264 MP4。这一目录最适合当前摄像头视觉初代版本。

  01_fall_positive
    数据集提供的原始跌倒视频。画面左侧是深度图，右侧是 RGB 彩色视频。

  02_daily_activity_negative
    数据集提供的原始日常活动视频。画面左侧是深度图，右侧是 RGB 彩色视频。

  ground_truth.csv
    每段“可直接上传版”视频的类别、预期结果、时长和来源清单。

三、数据来源与许可

数据集：UR Fall Detection Dataset（URFD）
发布单位：University of Rzeszow / University of Rzeszów
官方页面：https://fenix.ur.edu.pl/~mkepski/ds/uf.html

许可：Creative Commons Attribution-NonCommercial-ShareAlike 4.0
（CC BY-NC-SA 4.0），仅用于非商业学术研究。使用或展示时请保留来源，
如用于商业项目，应先联系数据集作者。

建议引用：
Bogdan Kwolek, Michal Kepski, Human fall detection on embedded platform
using depth maps and wireless accelerometer, Computer Methods and Programs
in Biomedicine, 117(3), 2014, 489-501.

四、重要说明

1. 这些是健康受试者在受控环境中模拟的跌倒，不是老年人的真实意外跌倒。
2. 这批文件适合验证上传、解码、YOLO 姿态识别和告警流程，不足以单独证明
   系统达到临床、医疗器械或真实养老环境的准确率要求。
3. 正例未告警可能说明姿态阈值、时序状态机、机位或模型仍需调整；负例告警
   则属于误报，应记录并用于优化。
4. 请勿让老年人或无保护人员自行模拟跌倒。后续自采数据应由健康成年人在
   软垫、护具和保护人员监督下完成，并取得肖像与研究知情同意。

五、已完成的检查

  - 12 段可直接上传视频均可完整打开并读取末帧。
  - 格式均为 H.264 MP4、640×480、30 FPS。
  - 使用项目内 yolo11n-pose 模型抽帧检查，12 段均检测到人体姿态。
