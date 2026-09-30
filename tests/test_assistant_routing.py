import asyncio
import json

import httpx
import pytest

from services import ai_assistant as ai
from services import knowledge as kb
from services.assistant_routing import Route, local_route, parse_route, retrieval_question, routing_history
from services.knowledge_import import save_document


def run_chat(question, history=None, use_knowledge=True, context='', consent=False):
    async def run():
        return [item async for item in ai.chat_auto(question, history or [], context, consent, use_knowledge)]
    return asyncio.run(run())


@pytest.mark.parametrize('question,intent', [('你好', 'general'), ('头痛用英语怎么说', 'general'),
    ('我有点头疼', 'symptom'), ('吃什么对眼睛好', 'health'), ('怎么导出记录', 'platform'),
    ('请总结这条选中记录，说明记录事实与不能确定的部分。', 'result')])
def test_explicit_intents_ignore_old_topic(question, intent):
    assert local_route(question, [{'role': 'user', 'content': '我头痛'}]).intents == (intent,)


@pytest.mark.parametrize('raw', ['{}', 'null', '{"intents":["execute"],"followup":false}',
    '{"intents":["general"],"followup":"false"}',
    '{"intents":["general"],"followup":false,"command":"delete"}',
    '{"intents":["clarify","health"],"followup":false}',
    '{"intents":["symptom","symptom"],"followup":false}'])
def test_invalid_classifications_rejected(raw):
    with pytest.raises(ValueError):
        parse_route(raw)


def mock_http(monkeypatch, respond):
    monkeypatch.setattr(ai, 'load_settings', lambda: {'key': 'TEST-SECRET', 'model': 'deepseek-v4-flash'})
    client = httpx.AsyncClient
    def make(**kwargs):
        assert kwargs['trust_env'] is False and kwargs['follow_redirects'] is False
        return client(transport=httpx.MockTransport(respond), **kwargs)
    monkeypatch.setattr(ai.httpx, 'AsyncClient', make)


def test_classifier_context_payload_and_schema(monkeypatch):
    def respond(request):
        body = json.loads(request.content)
        assert body['response_format'] == {'type': 'json_object'}
        assert body['max_tokens'] == 180 and body['stream'] is False
        assert 'tools' not in body
        assert 'SNAPSHOT-SECRET' not in request.content.decode()
        assert '已经两天了' in request.content.decode()
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
            'message': {'content': '{"intents":["symptom"],"followup":true}'}}]})
    mock_http(monkeypatch, respond)
    route = asyncio.run(ai.classify_question('已经两天了', [{'role': 'user',
        'content': '我头痛 以下是我确认发送的检测摘要SNAPSHOT-SECRET'}]))
    assert route == Route(('symptom',), True)


@pytest.mark.parametrize('packet', [{'choices': []}, {'choices': [{'finish_reason': 'length',
    'message': {'content': '{"intents":["health"],"followup":false}'}}]},
    {'choices': [{'finish_reason': 'stop', 'message': {'content': ''}}]}])
def test_empty_or_truncated_classification_fails_closed(monkeypatch, packet):
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=packet))
    assert asyncio.run(ai.classify_question('帮我看看', [])) is None


def test_classifier_errors_sanitized(monkeypatch):
    mock_http(monkeypatch, lambda request: httpx.Response(401, text='TEST-SECRET'))
    output = run_chat('帮我看看')[-1]
    assert '密钥无效' in output[0][-1]['content']
    assert 'TEST-SECRET' not in repr(output) and output[1] == []


def test_classifier_timeout_and_cancellation(monkeypatch):
    def timeout(request):
        raise httpx.ReadTimeout('TEST-SECRET')
    mock_http(monkeypatch, timeout)
    assert asyncio.run(ai.classify_question('已经两天了', [])) is None
    closed = []
    async def slow(question, history):
        try:
            await asyncio.sleep(60)
        finally:
            closed.append(True)
    monkeypatch.setattr(ai, 'classify_question', slow)
    async def run():
        async def consume():
            _ = [item async for item in ai.chat_auto('继续说', [], '', False, True)]
        task = asyncio.create_task(consume())
        await asyncio.sleep(.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
    assert closed


def test_urgent_precedes_credentials_classifier_and_rag(monkeypatch):
    monkeypatch.setattr(ai, 'load_settings', lambda: pytest.fail('no credentials'))
    monkeypatch.setattr(kb, 'store', lambda: pytest.fail('no RAG'))
    async def fail(*args):
        pytest.fail('no classifier')
    monkeypatch.setattr(ai, 'classify_question', fail)
    result = run_chat('刚撞到头，现在头疼')[-1]
    assert '120' in result[0][-1]['content'] and result[4] == '紧急提示'


def test_general_skips_rag_and_keeps_history(monkeypatch):
    monkeypatch.setattr(kb, 'store', lambda: pytest.fail('general must not search'))
    async def answer(messages):
        assert '健康咨询模式' not in messages[0]['content']
        assert messages[-1]['content'] == '头痛用英语怎么说'
        yield 'headache'
    monkeypatch.setattr(ai, 'stream_answer', answer)
    history = [{'role': 'user', 'content': '我有点头痛'}, {'role': 'assistant', 'content': '持续多久了？'}]
    result = run_chat('头痛用英语怎么说', history)[-1]
    assert result[1][:2] == history and result[4] == '日常问答'
    assert '未检索' in result[3]


def test_followup_and_topic_change_retrieval_scope(monkeypatch):
    seen = []
    class Store:
        def search(self, question, history, scope):
            seen.append((question, history, scope))
            return [], '没有匹配'
    monkeypatch.setattr(kb, 'store', Store)
    async def answer(messages):
        yield '测试回复'
    monkeypatch.setattr(ai, 'stream_answer', answer)
    async def classify(question, history):
        return Route(('symptom',), True) if question == '已经两天了' else Route(('platform',))
    monkeypatch.setattr(ai, 'classify_question', classify)
    history = [{'role': 'user', 'content': '我有点头痛'}, {'role': 'assistant', 'content': '持续多久？'}]
    first = run_chat('已经两天了', history)[-1]
    assert '头痛' in seen[0][0] and '已经两天了' in seen[0][0] and seen[0][2] == 'health'
    second = run_chat('怎么导出记录', first[1])[-1]
    assert seen[1] == ('怎么导出记录', [], 'platform')
    assert second[1][:2] == history


def test_scope_filters_before_ranking_and_preserves_review(tmp_path, monkeypatch):
    store = kb.KnowledgeStore(tmp_path)
    monkeypatch.setattr(kb, 'encoder', lambda: (_ for _ in ()).throw(RuntimeError('offline')))
    row = [dict(topic='头痛', section='概述', body='头痛相关资料', page_start=1, page_end=1)]
    save_document(store, '平台', 'v1', 1, 'platform', row, {}, review='approved')
    health = save_document(store, '科普', 'v1', 1, 'sourced', row, {}, review='sourced')
    save_document(store, 'PDF待审核', 'v1', 1, 'user', row, {})
    assert {h['name'] for h in store.search('头痛', scope='health')[0]} == {'科普'}
    assert {h['name'] for h in store.search('头痛', scope='platform')[0]} == {'平台'}
    store.set_enabled(health, False)
    assert not store.search('头痛', scope='health')[0]


def test_health_disabled_knowledge_no_stale_sources_and_no_snapshot_without_consent(monkeypatch):
    monkeypatch.setattr(kb, 'store', lambda: pytest.fail('knowledge disabled'))
    async def answer(messages):
        assert '健康咨询模式' in messages[0]['content']
        assert '不提供个体化处方' in messages[0]['content']
        assert 'PRIVATE' not in json.dumps(messages)
        assert 'NEI眼健康科普' not in json.dumps(messages, ensure_ascii=False)
        yield '测试科普'
    monkeypatch.setattr(ai, 'stream_answer', answer)
    result = run_chat('吃什么对眼睛好', use_knowledge=False, context='PRIVATE')[-1]
    assert result[4] == '健康科普' and '未使用' in result[3]


def test_mixed_request_health_first_and_confirmed_summary(monkeypatch):
    route = parse_route('{"intents":["result","symptom"],"followup":false}')
    assert route.intents[0] == 'symptom' and route.scope == 'all'
    async def classify(*args):
        return route
    async def answer(messages):
        assert '混合任务先症状咨询' in messages[0]['content']
        assert '跌倒事件证据分' in messages[-1]['content']
        yield '测试混合回复'
    monkeypatch.setattr(ai, 'classify_question', classify)
    monkeypatch.setattr(ai, 'stream_answer', answer)
    result = run_chat('刚跌倒腿疼，解释分数', use_knowledge=False,
                      context='跌倒事件证据分=30；前置风险分=20', consent=True)[-1]
    assert result[4].startswith('症状咨询')


def test_failed_classification_does_not_retrieve_or_generate(monkeypatch):
    async def classify(*args):
        return None
    monkeypatch.setattr(ai, 'classify_question', classify)
    monkeypatch.setattr(kb, 'store', lambda: pytest.fail('no retrieval'))
    monkeypatch.setattr(ai, 'stream_answer', lambda *args: pytest.fail('no generation'))
    history = [{'role': 'user', 'content': '你好'}, {'role': 'assistant', 'content': '你好'}]
    out = run_chat('这个呢', history)[-1]
    assert out[1] == history and '未能判断' in out[0][-1]['content']


def test_health_shortcut_clarifies_without_clearing_or_network(monkeypatch):
    monkeypatch.setattr(ai, 'load_settings', lambda: pytest.fail('no network'))
    history = [{'role': 'user', 'content': '你好'}, {'role': 'assistant', 'content': '你好'}]
    result = run_chat('我想咨询一个健康问题。', history)[-1]
    assert result[1][:2] == history and '请描述' in result[0][-1]['content']


def test_unified_ui_wiring_and_no_mode_reset():
    import app
    ui = app.build_app()
    config = ui.get_config_file()
    ids = {c['props'].get('elem_id') for c in config['components']}
    assert 'ai-mode' not in ids and {'ai-route-label', 'ai-shortcuts', 'muan-ai-panel'} <= ids
    callback = next(fn for fn in ui.fns.values() if fn.fn == ai.chat_auto)
    assert len(callback.inputs) == 5 and len(callback.outputs) == 5
