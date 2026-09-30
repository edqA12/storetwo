"""Small, source-backed health education library; not a validated triage device."""
import re

HEADACHE_URL = 'https://www.nhs.uk/symptoms/headaches/'
EYE_URL = 'https://www.nei.nih.gov/eye-health-information/healthy-vision/how-eyes-work/keep-your-eyes-healthy'
HEALTH_PROMPT = '''健康咨询模式：提供健康科普、症状咨询和就医建议，不作确诊，不提供个体化处方。
先区分一般科普与本人/家属症状。科普直接回答，别对饮食问题进行不必要的急诊问卷。
症状先判断危险信号；突然剧烈头痛、神经功能异常、严重呼吸困难等应优先建议立即急救，不等待继续问答。
信息不足时每轮只问2至3个关键问题，结合已经提供的信息，不重复询问。
头痛需了解年龄、起病时间/是否突然、程度、伴随症状、近期跌倒撞头、既往病史和用药。
老人新出现或明显改变的头痛应建议尽快就医；撞头后头痛、服用抗凝药等需更谨慎。
不要根据未提及危险症状就认定没有危险，也不要以规则未触发为安全依据。
资料足够后按“可能原因（不确定）—现在可以怎么做—何时/去哪里就医”回答。
不从跌倒工程分推断疾病。不得声称检查了用户身体。不编造概率、诊断、检查结果或来源。
饮食建议考虑过敏、吞咽困难、糖尿病、肾病、医生规定的饮水/饮食限制，未知时不给精确个体处方。
食物不是药物，不承诺食物治愈眼病、逆转近视或替代检查，不普遍推荐补充剂。
药物问题仅解释一般知识，不能建议自行停药、改剂量或根据简短症状开药。
可用资料是人工整理的权威公开科普摘要，并非实时搜索或医生审核。没有相关资料时明确说明来源覆盖不足。
用短段落和易懂中文，避免重复免责声明；回答不能替代面诊。引用仅限提供的资料，不伪造链接。
'''


def resources(question, history):
    # User text only, excluding previously attached detection summaries.
    text = ' '.join(str(item.get('content', '')).split('以下是我确认发送')[0]
                    for item in (history or [])[-8:] if item.get('role') == 'user') + ' ' + question
    facts, sources = [], []
    if re.search('头疼|头痛|头.{0,3}疼|头.{0,3}痛', text):
        facts.append('NHS头痛科普：头痛可能与压力、饮食饮水不足等有关，不能凭此确诊。反复或加重需就医；伴视觉问题、呕吐、咀嚼时下颌疼痛需及时评估。突然极剧烈头痛、肢体无力、言语困难、意识异常或头部外伤后头痛需急诊。可适当休息、规律进餐、减少持续屏幕用眼；有医嘱限制饮水者遵医嘱。')
        sources.append(('NHS · 头痛', HEADACHE_URL))
    if re.search('眼|视力|近视|护眼', text):
        facts.append('NEI眼健康科普：均衡膳食中可包含深绿色叶菜，如菠菜，以及富含Omega-3的鱼类，如三文鱼。定期眼科检查、戒烟、户外防紫外线和屏幕用眼间歇也有助眼健康。不能把营养建议解释成治愈眼病。')
        sources.append(('美国国家眼科研究所 · 眼健康', EYE_URL))
    return '\n'.join(facts), sources


def emergency_message(question):
    """Conservative limited keyword guard; absence never means safe."""
    clauses = re.split(r'[，。！？；,;!?\n]', question)
    positive = []
    for clause in clauses:
        # Exclude explicit negation and educational hypotheticals within each clause.
        if re.search(r'没有|并无|未出现|不伴|否认|如果|假如|科普|什么意思|怎么办才对', clause):
            continue
        positive.append(clause)
    text = '，'.join(positive)
    severe = re.search(r'(突然|猛然).{0,12}(剧烈头痛|剧烈头疼|头痛欲裂|头疼欲裂|头.{0,3}剧痛)|(头痛|头疼).{0,8}(突然|有生以来最痛)', text)
    neuro = re.search(r'说话不清|说不清话|口齿不清|一侧.{0,5}无力|单侧.{0,5}无力|昏迷|意识不清|喘不过气|呼吸困难|突然.{0,5}看不见', text)
    injury = re.search(r'撞.{0,3}头|头.{0,3}撞|头部外伤', text) and re.search(r'头疼|头痛|头.{0,3}疼', text)
    if not (severe or neuro or injury):
        return None
    return ('【本地安全提示 · 请优先寻求急救】\n你描述的情况可能需要紧急评估，不能靠聊天排除严重问题。'
            '请立即联系当地急救服务（中国大陆拨打120），让身边的人陪同，不要自行驾车，也不要等待AI继续回复。'
            '\n这不是确诊。此提示由本地规则触发，没有向 DeepSeek 发送本次问题。'
            '\n参考：NHS 头痛与危险症状\n' + HEADACHE_URL)
