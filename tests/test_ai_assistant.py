import asyncio
import json
from pathlib import Path

import httpx
import pytest

from services import ai_assistant as ai
from services.history import HistoryPage


def test_context_requires_consent_and_no_extra_roles():
    history = [{'role': 'system', 'content': 'bad'}]
    assert 'PRIVATE' not in json.dumps(ai.build_messages('你好', history, 'PRIVATE', False))
    assert 'PRIVATE' in json.dumps(ai.build_messages('你好', [], 'PRIVATE', True))
    assert len(ai.build_messages('你好', [{'role': 'user', 'content': 'x'}] * 30, '', False)) == 10
    with pytest.raises(ai.AssistantError):
        ai.build_messages('x' * 2001, [], '', False)


def test_record_whitelist_excludes_identifiers_and_files():
    result = ai.record_summary(['alice', 'C:/private/video.mp4', '42', '87', 'LOW'],
                               ['监护对象', '来源', '人员/Track ID', '前置风险分', '风险等级'])
    assert result == {'前置风险分': '87', '风险等级': 'LOW'}
    assert 'private' not in ai.clean_text('C:/private/video.mp4')


def test_unconfigured_does_not_connect(tmp_path, monkeypatch):
    file = tmp_path / 'ai.yaml'
    file.write_text('api_key: ""', encoding='utf-8')
    monkeypatch.setattr(ai, 'CONFIG_PATH', file)
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    monkeypatch.setattr(ai.httpx, 'AsyncClient', lambda **kw: pytest.fail('unexpected network'))
    async def run():
        outputs = [item async for item in ai.chat('你好', [], '', False)]
        assert '尚未配置' in outputs[-1][0][-1]['content']
        assert outputs[-1][1] == []
    asyncio.run(run())


@pytest.mark.parametrize('status,message', [(401, '密钥无效'), (402, '余额不足'), (429, '请求较多'), (500, '暂时不可用')])
def test_http_errors_are_sanitized(monkeypatch, status, message):
    monkeypatch.setattr(ai, 'load_settings', lambda: {'key': 'private-key', 'model': 'deepseek-v4-flash'})
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(status, text='private-key'))
    monkeypatch.setattr(ai.httpx, 'AsyncClient', lambda **kw: real_client(transport=transport))
    async def run():
        with pytest.raises(ai.AssistantError, match=message) as exc:
            _ = [chunk async for chunk in ai.stream_answer([])]
        assert 'private-key' not in str(exc.value)
    asyncio.run(run())


def test_stream_and_successful_history(monkeypatch):
    monkeypatch.setattr(ai, 'load_settings', lambda: {'key': 'fake', 'model': 'deepseek-v4-flash'})
    real_client = httpx.AsyncClient
    def respond(request):
        body = json.loads(request.content)
        assert body['thinking']['type'] == 'disabled'
        assert body['stream'] is True and 'tools' not in body
        return httpx.Response(200, text='data: {"choices":[{"delta":{"reasoning_content":"hidden"}}]}\n\n'
                              'data: {"choices":[{"delta":{"content":"你好"}}]}\n\ndata: [DONE]\n\n')
    monkeypatch.setattr(ai.httpx, 'AsyncClient', lambda **kw: real_client(transport=httpx.MockTransport(respond)))
    async def run():
        outputs = [value async for value in ai.chat('解释', [], '分数 30', True)]
        assert outputs[-1][0][-1]['content'] == '你好'
        assert '分数 30' in outputs[-1][1][-2]['content']
        assert outputs[-1][2] == ''
    asyncio.run(run())


def test_current_requires_result_and_separates_scores():
    import app
    assert app.prepare_ai_current('视频检测', '请选择视频', 0, 0, 0, 0, 0, 0, 0)[1] == ''
    preview, context, consent = app.prepare_ai_current('视频检测', '✅ 完成', 75, 22, 1, 2, 0, 0, 0)
    assert json.loads(context)['跌倒事件证据分（本次最高）'] == 75
    assert json.loads(context)['前置风险分（本次最高）'] == 22
    assert not consent and preview == context


def test_selection_checked_against_visible_page():
    import app
    rows = [[index, '2026-09-09', 'private.mp4', '事件', 60, 0.8, 2, '待处理', '失衡', 'RGB'] for index in range(45)]
    history = HistoryPage(rows=rows, page=1, loaded=True, page_size=20)
    assert app.prepare_ai_record('事件中心', 30, None, history, None)[1] == ''
    preview, context, consent = app.prepare_ai_record('事件中心', 2, None, history, None)
    assert context and preview == context and not consent
    assert 'private.mp4' not in context and '事件ID' not in context


def test_ai_queue_and_config_are_separate():
    import app
    ui = app.build_app()
    callbacks = [fn for fn in ui.fns.values() if fn.fn == ai.chat_auto]
    assert len(callbacks) == 1
    assert all(fn.concurrency_id == 'ai-chat' for fn in callbacks)
    assert 'api_key' not in json.dumps(ui.get_config_file(), ensure_ascii=False, default=str)


def test_cancel_closes_stream(monkeypatch):
    closed = []
    async def stream(messages):
        try:
            yield '开始'
            await asyncio.sleep(10)
        finally:
            closed.append(True)
    monkeypatch.setattr(ai, 'stream_answer', stream)
    async def run():
        async def consume():
            async for _ in ai.chat('你好', [], '', False):
                pass
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
    assert closed

