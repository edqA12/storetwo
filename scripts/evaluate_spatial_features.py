"""Evaluate shadow 3D rules against annotated metric clips; never edit live config."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path


def evaluate(rows: list[dict], height: float, speed: float, duration: float) -> dict:
    tp = fp = fn = tn = 0
    for row in rows:
        predicted = (float(row['torso_height_m']) < height and
                     (float(row['descent_speed_m_s']) >= speed or float(row['low_duration_s']) >= duration))
        positive = row['label'] == 'fall_evidence'
        tp += predicted and positive
        fp += predicted and not positive
        fn += not predicted and positive
        tn += not predicted and not positive
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def run(source: Path, output: Path) -> None:
    import math
    with source.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('没有标注数据，不能调整三维阈值或融合权重')
    splits: dict[str, set[str]] = {}
    for row in rows:
        if row['source'] != 'astra_metric' or row['reviewed'] != 'true' or len(row['calibration_id']) != 64:
            raise ValueError('只接受经核对、带标定标识的Astra米制特征，不能用8位视频或模拟轨迹调参')
        if row['label'] not in {'fall_evidence', 'non_fall'} or row['split'] not in {'train', 'validation'}:
            raise ValueError('标签或数据划分无效')
        if not row['subject_id'] or not row['clip_id']:
            raise ValueError('缺少人员或片段标识')
        if any(not math.isfinite(float(row[k])) for k in ('torso_height_m', 'descent_speed_m_s', 'low_duration_s')):
            raise ValueError('存在不可用特征；请单独统计不可用率，不能当作零')
        for key in ('subject_id', 'clip_id'):
            splits.setdefault(key + ':' + row[key], set()).add(row['split'])
    if any(len(value) > 1 for value in splits.values()):
        raise ValueError('人员或片段跨训练/验证集，存在泄漏')
    train = [row for row in rows if row['split'] == 'train']
    validation = [row for row in rows if row['split'] == 'validation']
    for subset in (train, validation):
        if {row['label'] for row in subset} != {'fall_evidence', 'non_fall'}:
            raise ValueError('训练与验证集都需要正负标注；样本代表性还须人工审核')
    candidates = []
    for height, speed, duration in itertools.product((.35, .45, .55), (.3, .6, .9), (1., 2., 3.)):
        candidates.append({'height_m': height, 'speed_m_s': speed, 'duration_s': duration,
                           'train': evaluate(train, height, speed, duration)})
    selected = max(candidates, key=lambda item: item['train']['f1'] or 0)
    selected['validation'] = evaluate(validation, selected['height_m'], selected['speed_m_s'], selected['duration_s'])
    report = {'unit': 'labelled_frame', 'selected_on': 'train_only', 'selected': selected,
              'candidates': candidates, 'counts': {'train': len(train), 'validation': len(validation)},
              'applied_to_live': False, 'fusion_weights_changed': False,
              'limitations': '探索性逐帧三维规则评估；不是事件准确率。需事件级RGB/融合对照和独立人员测试后再人工审阅融合参数。'}
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)
