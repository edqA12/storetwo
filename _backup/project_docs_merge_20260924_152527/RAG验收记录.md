# RAG 知识库软件验收记录

> 历史记录说明（2026-09-13补充）：以下片段数量、审核状态与111项测试为初次导入／验收时的记录，不是实时数据库统计。文中模式切换属于旧版界面；当前统一聊天入口仅在切换知识库开关时清空会话。最新操作见 [RAG知识库使用说明](RAG知识库使用说明.md)。

- 本地片段总数：26168；已生成向量：26168。
- 自动测试：111 项通过（包括原有检测、历史记录、AI健康咨询与新增知识库检查）。
- 代表性检索样例：7/7 满足预期来源或无命中条件。该小样本不代表全面检索准确率、医学准确率或相对基线提升幅度。
- 浏览器：桌面及390px手机宽度验证管理入口、候选检索、原文引用、紧急提示跳过检索、模式切换。
- 未使用真实密钥发送测试问题；DeepSeek生成链路用离线替身测试，真实模型回复质量仍需持续人工评估。
- 两份用户PDF均为待核对状态。没有自动批准医疗内容，也没有声称完成临床验证。

## 导入报告

### NHS头痛科普（来源整理）

页数：0

```json
{
  "notice": "公开科普来源整理，非临床审核",
  "checked_at": "2026-09-09"
}
```

### NEI眼健康饮食（来源整理）

页数：0

```json
{
  "notice": "公开科普来源整理，非临床审核",
  "checked_at": "2026-09-09"
}
```

### 暮安智护平台说明

页数：0

```json
{}
```

### 医疗健康.pdf

页数：5

```json
{
  "pages": 5,
  "blank_pages": 0,
  "format": "sections",
  "records": 0,
  "missing_names": 0,
  "removed_noise": 0,
  "excluded_fields": [
    "drug_detail",
    "common_drug",
    "recommand_drug",
    "cure_way",
    "get_prob",
    "cured_prob",
    "cost_money",
    "do_eat",
    "not_eat",
    "recommand_eat"
  ],
  "chunks": 19
}
```

### 医疗疾病知识库.pdf

页数：6856

```json
{
  "pages": 6856,
  "blank_pages": 0,
  "format": "disease_records",
  "records": 8806,
  "missing_names": 0,
  "removed_noise": 492,
  "excluded_fields": [
    "drug_detail",
    "common_drug",
    "recommand_drug",
    "cure_way",
    "get_prob",
    "cured_prob",
    "cost_money",
    "do_eat",
    "not_eat",
    "recommand_eat"
  ],
  "chunks": 26144
}
```

