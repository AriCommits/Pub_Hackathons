"""Validation-only threshold selection for first-hop routing."""
from __future__ import annotations

import math
from collections import defaultdict


def calibrate(scores, *, split, min_coverage, score_grid, margin_grid):
    """Select accuracy, then margin, then score; retain the entire coverage curve.

    Each record has gold, predicted, score, margin and family. Unrouted records
    remain in the coverage denominator and do not contribute to routed accuracy.
    """
    if split != 'val':
        raise ValueError('threshold calibration requires the validation split')
    if not scores or not 0 < min_coverage <= 1 or not score_grid or not margin_grid:
        raise ValueError('invalid calibration data, grid, or coverage')
    for row in scores:
        if not row.get('gold') or not row.get('predicted') or not row.get('family'):
            raise ValueError('calibration records require gold, predicted and family')
        if not all(math.isfinite(float(row[k])) for k in ('score', 'margin')):
            raise ValueError('non-finite routing score')
    if not all(math.isfinite(float(x)) for x in [*score_grid, *margin_grid]):
        raise ValueError('non-finite threshold')
    curve = []
    for margin in margin_grid:
        for score in score_grid:
            selected = [r for r in scores if r['score'] >= score and r['margin'] >= margin]
            n, routed = len(scores), len(selected)
            correct = sum(r['gold'] == r['predicted'] for r in selected)
            point = {'min_score': float(score), 'min_margin': float(margin),
                     'routed': routed, 'total': n, 'coverage': routed / n,
                     'accuracy': correct / routed if routed else None,
                     'risk': (routed - correct) / routed if routed else None}
            curve.append(point)
    eligible = [p for p in curve if p['coverage'] >= min_coverage and p['routed']]
    best = max(eligible, key=lambda p: (p['accuracy'], p['min_margin'], p['min_score'])) if eligible else None
    families = defaultdict(lambda: {'total': 0, 'routed': 0, 'correct': 0, 'confusion': {}})
    if best:
        for row in scores:
            family = families[row['family']]
            family['total'] += 1
            if row['score'] >= best['min_score'] and row['margin'] >= best['min_margin']:
                family['routed'] += 1
                family['correct'] += int(row['gold'] == row['predicted'])
                key = row['gold'] + ' -> ' + row['predicted']
                family['confusion'][key] = family['confusion'].get(key, 0) + 1
    return {'split': split, 'minimum_coverage': min_coverage, 'selected': best,
            'risk_coverage_curve': curve, 'families': dict(families)}
