import asyncio
import pytest
from services import ai_assistant as ai
from services.health_guidance import emergency_message, resources


@pytest.mark.parametrize('text', ['我突然剧烈头痛', '我现在口齿不清，一侧手臂无力', '刚撞到头，现在头疼'])
def test_clear_red_flags(text):
    assert '120' in emergency_message(text)


@pytest.mark.parametrize('text', ['吃什么对眼睛好', '我有点头疼', '没有口齿不清', '如果突然剧烈头痛会怎样'])
def test_no_unconditional_emergency(text):
    assert emergency_message(text) is None


def test_health_grounding_and_default_mode():
    normal = ai.build_messages('眼睛', [], '', False)[0]['content']
    health = ai.build_messages('吃什么对眼睛好', [], '', False, '健康咨询')[0]['content']
    assert 'NEI眼健康科普' not in normal
    assert 'NEI眼健康科普' in health and '不提供个体化处方' in health
    assert '本地资料库未覆盖' in ai.build_messages('膝盖疼', [], '', False, '健康咨询')[0]['content']


def test_followup_sources_use_user_history_only():
    facts, sources = resources('那菠菜呢', [{'role': 'user', 'content': '吃什么对眼睛好'}])
    assert sources and '叶菜' in facts
    assert not resources('你好', [{'role': 'assistant', 'content': '眼睛'}])[1]


def test_urgent_works_without_key_or_network(monkeypatch):
    monkeypatch.setattr(ai, 'load_settings', lambda: pytest.fail('should not read credentials'))
    async def run():
        items = [item async for item in ai.chat('我突然剧烈头痛', [], '', False, '健康咨询')]
        assert '本地安全提示' in items[-1][0][-1]['content']
    asyncio.run(run())


def test_health_response_appends_real_sources(monkeypatch):
    async def mock(messages):
        assert 'NEI眼健康科普' in messages[0]['content']
        yield '均衡饮食中可以包括菠菜。'
    monkeypatch.setattr(ai, 'stream_answer', mock)
    async def run():
        items = [item async for item in ai.chat('吃什么对眼睛好', [], '', False, '健康咨询')]
        assert 'nei.nih.gov' in items[-1][0][-1]['content']
        assert items[-1][1][-1]['content'] == items[-1][0][-1]['content']
    asyncio.run(run())
