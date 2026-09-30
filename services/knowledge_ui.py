from pathlib import Path
import gradio as gr
from services.knowledge import store, citations_text
from services.knowledge_import import import_file


def catalog():
    documents = store().documents()
    rows = [[d['name'], d['pages'], d['chunks'], d['vectors'], d['usable'] or 0,
             '启用' if d['enabled'] else '停用', d['version'][:12]] for d in documents]
    choices = [(d['name'] + ' · ' + d['version'][:8], d['id']) for d in documents]
    return rows, gr.update(choices=choices, value=choices[0][1] if choices else None)


def ingest(file, progress=gr.Progress()):
    if not file:
        return *catalog(), '请先上传 PDF、TXT 或 Markdown 文件。'
    try:
        import_file(store(), Path(file), lambda message: progress(.15, desc=message))
        store().embed_missing(lambda message: progress(.65, desc=message))
        return *catalog(), '导入及索引完成。新增内容为待审核状态，仅可在本地检索预览。'
    except Exception:
        return *catalog(), '导入或索引未完成。已完成的片段会保留，可点击“补齐向量索引”重试；文件须有可提取文字且不超过100 MB。'


def rebuild(progress=gr.Progress()):
    try:
        count = store().embed_missing(lambda message: progress(.5, desc=message))
        return *catalog(), f'索引完成，本次补齐 {count} 个向量。'
    except Exception:
        return *catalog(), '向量模型暂不可用，请检查本地模型和运行库；已导入文字仍可用关键词预览。'


def enable_document(doc_id, enabled):
    if not doc_id:
        return *catalog(), '请先选择文档。'
    store().set_enabled(doc_id, enabled)
    return *catalog(), ('文档已启用。待审核片段仍不会参与回答。' if enabled else '文档已停用，不再参与新的检索回答。')


def preview(query, doc_id, pending):
    if not query.strip():
        return [], '请输入检索问题。', None, '', '', False, None
    hits, status = store().search(query, preview=bool(pending), doc_id=doc_id, limit=12)
    rows = [[h['id'], h['name'], h['topic'], h['section'],
             f'{h["page_start"]}—{h["page_end"]}', h['review'], h['relevance']] for h in hits]
    return rows, status + '；此处仅本地预览，不调用 DeepSeek。', None, '', '', False, None


def select_chunk(evt: gr.SelectData):
    row = getattr(evt, 'row_value', None)
    chunk = store().chunk(str(row[0])) if isinstance(row, (list, tuple)) and row else None
    if not chunk:
        return None, '请重新选择片段。', '', False, None
    original = None
    if chunk['original']:
        base = (store().root / 'originals').resolve()
        path = (base / chunk['original']).resolve()
        if path.is_relative_to(base) and path.is_file():
            original = str(path)
    text = citations_text([chunk], '选中原文；待审核内容不可直接作为健康建议')
    return chunk['id'], text, chunk['reviewer_note'], False, original


def review_chunk(chunk_id, note, confirmed, approved):
    if not chunk_id:
        return *catalog(), '请先检索并点击一条片段。'
    if approved and not confirmed:
        return *catalog(), '请核对原文并填写依据后勾选确认；一次只处理所选片段。'
    try:
        store().review(chunk_id, approved, note)
        return *catalog(), '所选片段已更新；重新检索可查看新状态。核对不等同于临床验证。'
    except ValueError as exc:
        return *catalog(), str(exc)


def mount():
    with gr.Column(elem_id='muan-kb-panel'):
        gr.HTML('<div class="ai-heading"><div><strong>暮安知识库</strong><small>本地检索 · 原文可追溯</small></div>'
                '<button type="button" id="muan-kb-close" aria-label="关闭知识库">×</button></div>')
        gr.Markdown('上传资料默认待审核。**启用文档不等于通过审核**；只有已核对片段或带权威出处的整理资料会参与回答。'
                    '下方检索预览可包含待审核资料，不会发送给 DeepSeek。全库仅当前本机使用。')
        with gr.Row(elem_id='kb-layout'):
            with gr.Column(scale=3, elem_id='kb-library'):
                table = gr.Dataframe(headers=['文档','页数','片段','向量','可用片段','状态','版本'], value=[], interactive=False, elem_id='kb-catalog')
                with gr.Row():
                    refresh = gr.Button('刷新知识库', elem_id='kb-refresh')
                    document = gr.Dropdown(choices=[], label='选择要管理或检索的文档', filterable=False, elem_id='kb-document')
                    on = gr.Button('启用文档')
                    off = gr.Button('停用文档')
                with gr.Accordion('导入或更新资料', open=False):
                    upload = gr.File(label='上传 PDF / TXT / Markdown（最大100 MB）', file_types=['.pdf','.txt','.md'], type='filepath')
                    with gr.Row():
                        submit = gr.Button('导入并建立索引', variant='primary')
                        resume = gr.Button('补齐向量索引')
                    gr.Markdown('大文件导入可能较久，请在停止视频检测后操作。重复导入相同文件不重复建库；同名新版本会停用旧版本，并重新进入待审核状态。')
            with gr.Column(scale=5, elem_id='kb-reader'):
                with gr.Row():
                    query = gr.Textbox(label='本地检索预览', placeholder='例如：头痛、血压、眼健康饮食', elem_id='kb-query')
                    search = gr.Button('检索预览', variant='primary', elem_id='kb-search')
                pending = gr.Checkbox(label='包含待审核／停用资料（仅本地预览）', value=True)
                status = gr.Markdown('点击刷新查看已导入资料。', elem_id='kb-status')
                results = gr.Dataframe(headers=['片段ID','文档','主题','字段','原始页码','核对状态','排序分'], value=[], interactive=False, elem_id='kb-results')
                selected = gr.State(None)
                excerpt = gr.Textbox(label='引用原文', lines=7, interactive=False)
                original = gr.File(label='下载选中文档原件', interactive=False, elem_id='kb-original')
                with gr.Accordion('核对所选片段', open=False):
                    note = gr.Textbox(label='核对依据／权威来源链接', placeholder='写明核对依据；不建议凭文件名称判断医学可靠性。')
                    confirm = gr.Checkbox(label='已核对所选片段，确认可作为科普参考', value=False)
                    with gr.Row():
                        approve = gr.Button('允许所选片段参与回答')
                        revoke = gr.Button('撤回所选片段')
        refresh.click(catalog, outputs=[table,document], queue=False)
        for button, enabled in [(on, True),(off, False)]:
            button.click(lambda doc_id, enabled=enabled: enable_document(doc_id, enabled), inputs=document,
                         outputs=[table,document,status], queue=False)
        submit.click(ingest, inputs=upload, outputs=[table,document,status], concurrency_id='knowledge-import', concurrency_limit=1)
        resume.click(rebuild, outputs=[table,document,status], concurrency_id='knowledge-import', concurrency_limit=1)
        search.click(preview, inputs=[query,document,pending], outputs=[results,status,selected,excerpt,note,confirm,original], concurrency_id='knowledge-search', concurrency_limit=1)
        results.select(select_chunk, outputs=[selected,excerpt,note,confirm,original], queue=False)
        document.change(lambda: ([], None, '', '', False, None),
                        outputs=[results,selected,excerpt,note,confirm,original], queue=False)
        approve.click(lambda cid,note,confirmed: review_chunk(cid,note,confirmed,True), inputs=[selected,note,confirm], outputs=[table,document,status], queue=False)
        revoke.click(lambda cid: review_chunk(cid,'',False,False), inputs=selected, outputs=[table,document,status], queue=False)
