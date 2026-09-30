import csv
import json
import pytest
from scripts.evaluate_spatial_features import run


def dataset(tmp_path, mutate=None):
    rows = []
    for split in ('train', 'validation'):
        for label, height in [('fall_evidence', .2), ('non_fall', 1.2)]:
            rows.append(dict(source='astra_metric', reviewed='true', calibration_id='a'*64,
                             split=split, subject_id=split, clip_id=split+label, label=label,
                             torso_height_m=height, descent_speed_m_s=.7, low_duration_s=3))
    if mutate:
        mutate(rows)
    path = tmp_path / 'features.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_shadow_evaluation_does_not_apply_parameters(tmp_path):
    output = tmp_path / 'result.json'
    run(dataset(tmp_path), output)
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['selected']['validation']['tp'] == 1
    assert report['selected']['validation']['fp'] == 0
    assert not report['applied_to_live'] and not report['fusion_weights_changed']
    assert report['selected_on'] == 'train_only'


@pytest.mark.parametrize('mutation', [
    lambda rows: rows[-1].update(source='rgb_video'),
    lambda rows: rows[-1].update(subject_id='train'),
    lambda rows: rows[-1].update(reviewed='false'),
    lambda rows: rows[-1].update(torso_height_m='nan'),
])
def test_reject_unusable_annotations(tmp_path, mutation):
    with pytest.raises(ValueError):
        run(dataset(tmp_path, mutation), tmp_path / 'no_report.json')
