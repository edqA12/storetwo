"""PDF/text ingestion with stable IDs, page provenance and explicit review state."""
from __future__ import annotations

import bisect
import hashlib
import json
import re
import shutil
from contextlib import closing
from datetime import datetime
from pathlib import Path

from services.knowledge import KnowledgeStore, clean, tokens


def split_text(text: str, size: int = 380) -> list[str]:
    text = clean(text)
    # Keep decimal points and units. Oversized sentences are hard-bounded, not silently truncated.
    sentences = re.split(r'(?<=[。！？；])|(?<=[.!?;])\s+', text)
    pieces, current = [], ''
    for sentence in sentences:
        sentence = sentence.strip()
        while len(sentence) > size:
            if current:
                pieces.append(current)
                current = ''
            pieces.append(sentence[:size])
            sentence = sentence[size:]
        if current and len(current) + len(sentence) > size:
            pieces.append(current)
            current = ''
        current += sentence
    if current:
        pieces.append(current)
    return [p for p in pieces if len(p) >= 10]


def read_pages(path: Path, progress=lambda message: None) -> list[str]:
    if path.suffix.lower() == '.pdf':
        import pypdfium2 as pdfium
        result = []
        with closing(pdfium.PdfDocument(str(path))) as document:
            for i in range(len(document)):
                page = document[i]
                textpage = page.get_textpage()
                result.append(textpage.get_text_bounded().replace('\r\n', '\n'))
                textpage.close()
                page.close()
                if i % 500 == 0:
                    progress(f'读取PDF {i + 1}/{len(document)}页')
        return result
    if path.suffix.lower() in {'.txt', '.md'}:
        return [path.read_text(encoding='utf-8-sig')]
    raise ValueError('仅支持 PDF、Markdown 和 UTF-8 TXT。')


def extract_chunks(pages: list[str]) -> tuple[list[dict], dict]:
    offsets, position = [], 0
    for page in pages:
        offsets.append(position)
        position += len(page) + 1
    text = '\n'.join(pages)
    starts = [m.start() for m in re.finditer(r'\{\s*"_id"\s*:', text)]
    report = {'pages': len(pages), 'blank_pages': sum(not page.strip() for page in pages),
              'format': 'disease_records' if starts else 'sections', 'records': len(starts),
              'missing_names': 0, 'removed_noise': 0,
              'excluded_fields': ['drug_detail', 'common_drug', 'recommand_drug', 'cure_way',
                                  'get_prob', 'cured_prob', 'cost_money', 'do_eat', 'not_eat', 'recommand_eat']}
    rows = []
    if starts:
        # Recover field values separately, so PDF line wraps need not form valid JSON.
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else len(text)
            record = text[start:end]
            name_match = re.search(r'"name"\s*:\s*"((?:\\.|[^"\\])*)"', record, re.S)
            if not name_match:
                report['missing_names'] += 1
                continue
            name = clean(name_match.group(1))
            for field, label in [('desc', '概述'), ('symptom', '症状条目'), ('cure_department', '就诊科室')]:
                pattern = r'"' + field + r'"\s*:\s*("(?:\\.|[^"\\])*"|\[[^\]]*\])'
                found = re.search(pattern, record, re.S)
                if not found:
                    continue
                raw = found.group(1)
                try:
                    value = json.loads(raw, strict=False)
                except (json.JSONDecodeError, ValueError):
                    value = raw.strip('"[]')
                if isinstance(value, list):
                    kept = []
                    for item in value:
                        item = clean(str(item))
                        if any(noise in item for noise in ['毓卓', '闫鹏辉', '驻站医']):
                            report['removed_noise'] += 1
                        else:
                            kept.append(item)
                    value = '、'.join(kept)
                # Start/end pages refer to complete source field, not fabricated exact line positions.
                first_page = bisect.bisect_right(offsets, start + found.start())
                last_page = bisect.bisect_right(offsets, start + found.end() - 1)
                body = clean(str(value))
                if label != '概述' and len(body) < 10:
                    body = name + '：' + body
                for piece in split_text(body):
                    rows.append({'topic': name, 'section': label, 'body': piece,
                                 'page_start': first_page, 'page_end': last_page})
    else:
        # Carry headings across page boundaries. The full section page range remains traceable.
        headings = list(re.finditer(r'(?m)^(?:#{1,4}\s+.+|\s*\d+\.\s*[^\n]{1,60}[（(][^\n]{1,60}[）)]\s*[：:]?)', text))
        if not headings:
            headings = []
            spans = [(0, len(text), '文档正文')]
        else:
            spans = [(m.start(), headings[i + 1].start() if i + 1 < len(headings) else len(text), clean(m.group()))
                     for i, m in enumerate(headings)]
        for start, end, title in spans:
            for piece in split_text(text[start:end]):
                rows.append({'topic': title, 'section': '科普说明', 'body': piece,
                             'page_start': bisect.bisect_right(offsets, start),
                             'page_end': bisect.bisect_right(offsets, max(start, end - 1))})
    report['chunks'] = len(rows)
    return rows, report


def save_document(store: KnowledgeStore, name: str, version: str, pages: int, kind: str,
                  rows: list[dict], report: dict, original: str = '', review: str = 'pending') -> str:
    doc_id = hashlib.sha256((name + ':' + version).encode()).hexdigest()[:24]
    with store.connect() as db:
        if db.execute('SELECT id FROM documents WHERE id=?', (doc_id,)).fetchone():
            return doc_id
        # New versions do not inherit content approval. Old versions are deactivated atomically.
        db.execute('UPDATE documents SET enabled=0 WHERE name=?', (name,))
        db.execute('INSERT INTO documents VALUES (?,?,?,?,?,1,?,?,?)',
                   (doc_id, name, version, pages, kind, original, datetime.now().isoformat(timespec='seconds'), json.dumps(report, ensure_ascii=False)))
        for i, row in enumerate(rows):
            chunk_id = hashlib.sha256((doc_id + str(i) + row['body']).encode()).hexdigest()[:24]
            db.execute('INSERT INTO chunks (id,doc_id,topic,section,body,page_start,page_end,review) VALUES (?,?,?,?,?,?,?,?)',
                       (chunk_id, doc_id, row['topic'], row['section'], row['body'], row['page_start'], row['page_end'], review))
            term_text = ' '.join(tokens(row['topic'] + ' ' + row['topic'] + ' ' + row['section'] + ' ' + row['body']))
            db.execute('INSERT INTO chunks_fts VALUES (?,?)', (chunk_id, term_text))
    return doc_id


def import_file(store: KnowledgeStore, path: Path, progress=lambda message: None) -> str:
    path = Path(path)
    if path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError('单个文件最大100 MB，请先拆分。')
    version = hashlib.sha256(path.read_bytes()).hexdigest()
    existing = next((d for d in store.documents() if d['name'] == path.name and d['version'] == version), None)
    if existing:
        progress('相同文件已导入，保留原有核对状态。')
        return existing['id']
    pages = read_pages(path, progress)
    rows, report = extract_chunks(pages)
    if not rows:
        raise ValueError('未提取到可用文字。扫描件请先完成 OCR，不能将空页当作成功导入。')
    originals = store.root / 'originals'
    originals.mkdir(exist_ok=True)
    stored_name = version[:24] + path.suffix.lower()
    destination = originals / stored_name
    if not destination.exists():
        shutil.copy2(path, destination)
    doc_id = save_document(store, path.name, version, len(pages), 'user', rows, report, stored_name)
    (store.root / (doc_id + '.cleaned.jsonl')).write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows), encoding='utf-8')
    progress(f'导入完成：{len(pages)}页，{len(rows)}个候选片段；医学内容未审核。')
    return doc_id


def seed_sources(store: KnowledgeStore):
    from services.health_guidance import resources
    for query, name in [('头痛', 'NHS头痛科普（来源整理）'), ('眼睛', 'NEI眼健康饮食（来源整理）')]:
        body, sources = resources(query, [])
        rows = [{'topic': query + (' 眼健康 护眼 饮食 营养 菠菜' if query == '眼睛' else ' 头疼 脑袋疼'),
                 'section': '权威来源科普摘要', 'body': body + '\n来源：' + sources[0][1], 'page_start': 0, 'page_end': 0}]
        save_document(store, name, hashlib.sha256(body.encode()).hexdigest(), 0, 'sourced', rows,
                      {'notice': '公开科普来源整理，非临床审核', 'checked_at': '2026-09-09'}, review='sourced')
    rows = [
        {'topic': '视频检测 RGB 深度输入 上传视频', 'section': '使用方法', 'body': '视频检测支持普通RGB，以及同一时刻左侧深度图、右侧RGB的组合视频。实时摄像头仅RGB。上传分析前停止摄像头，选择对应输入模式，再点击开始检测。', 'page_start': 0, 'page_end': 0},
        {'topic': '跌倒事件证据分 前置风险分 评分 医学概率', 'section': '评分边界', 'body': 'Fall Event Score为跌倒事件证据分，Pre-Fall Risk Score为前置风险分，反映相对个人行为基线的变化；两者不同，均非医学概率。上传视频不更新持久个人基线。未检测到跌倒不等于身体安全。', 'page_start': 0, 'page_end': 0},
        {'topic': '历史记录 事件中心 翻页 导出 CSV', 'section': '历史查询', 'body': '事件中心浏览最近200条告警，风险历史浏览最近500条记录，支持每页20、50条或全部。CSV导出上限5000条。选择一行查看详情，翻页后需重新选择。', 'page_start': 0, 'page_end': 0}]
    save_document(store, '暮安智护平台说明', 'platform-20260910-v1', 0, 'platform', rows, {}, review='approved')
