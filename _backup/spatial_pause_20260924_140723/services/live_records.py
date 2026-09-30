"""Small, JSON-safe provenance snapshots shared by live UI and history."""
from __future__ import annotations

import json
from core.spatial import spatial_markdown
from typing import Any


def capture_context(result: Any, source: str, device: str) -> dict[str, Any]:
    diagnostics = result.diagnostics
    quality = {m.modality: round(float(m.quality), 3) for m in result.modalities}
    weights = dict(result.fusion.modality_weights) if result.fusion else {'vision': 1.0}
    used = [m.modality for m in result.modalities if m.available and weights.get(m.modality, 0) > 0]
    if not result.decisions:
        result_state = 'no_person'
    elif not diagnostics.get('rgb_result_usable', any(m.modality == 'vision' and m.available for m in result.modalities)):
        result_state = 'insufficient_quality'
    else:
        result_state = 'result'
    return {
        'schema': 1, 'source': source, 'device': device,
        'device_ids': diagnostics.get('device_ids', {}),
        'capture_session': diagnostics.get('capture_session'),
        'modalities_used': used, 'quality': quality, 'weights': weights,
        'fallback_reason': diagnostics.get('fallback_reason', ''),
        'depth_valid_ratio': diagnostics.get('depth_valid_ratio'),
        'pair_delta_s': diagnostics.get('software_pair_delta_s'),
        'result_state': result_state,
        'frame_sequence': diagnostics.get('frame_sequence'),
        'frame_time_s': diagnostics.get('frame_time_s'),
        'spatial_registration': diagnostics.get('spatial_registration', False),
        'body_3d': diagnostics.get('body_3d', {}),
    }


def context_values(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            value = {}
    if not isinstance(value, dict) or not value:
        return ['未记录', '未记录', '未记录', '未记录']
    labels = {'vision': 'RGB', 'depth': '深度'}
    return [str(value.get('device') or '未记录'),
            '＋'.join(labels.get(m, m) for m in value.get('modalities_used', [])) or '无有效模态',
            ' / '.join(f'{labels.get(k, k)} {v:.2f}' for k, v in value.get('quality', {}).items()),
            str(value.get('fallback_reason') or '无')]


def context_markdown(value: Any) -> str:
    device, modes, quality, reason = context_values(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = {}
    spatial = spatial_markdown(value.get('body_3d')) if isinstance(value, dict) and value.get('body_3d') else ''
    return f'\n- 设备：{device}\n- 实际所用模态：{modes}\n- 模态质量：{quality}\n- 回退原因：{reason}\n' + spatial
