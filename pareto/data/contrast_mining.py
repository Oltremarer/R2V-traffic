from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Tuple

from pareto.constants import OBJECTIVE_NAMES


def _obj(record: Dict, name: str) -> float:
    return float(record["objective_values_norm"][name])


def _utility(record: Dict, w: Iterable[float]) -> float:
    return sum(float(weight) * _obj(record, name) for weight, name in zip(w, OBJECTIVE_NAMES))


def _label(a_score: float, b_score: float) -> int:
    return int(a_score > b_score)


def mine_efficiency_controlled_pairs(
    records: List[Dict],
    target_objective: str,
    n: int,
    eps_efficiency: float,
    margin_target: float,
) -> List[Dict]:
    pairs = []
    for a, b in combinations(records, 2):
        eff_gap = abs(_obj(a, "efficiency") - _obj(b, "efficiency"))
        target_gap = _obj(a, target_objective) - _obj(b, target_objective)
        if eff_gap <= eps_efficiency and abs(target_gap) >= margin_target:
            pairs.append({
                "a_id": a["sample_id"],
                "b_id": b["sample_id"],
                "label": int(target_gap > 0),
                "objective": target_objective,
                "sampling_strategy": f"eff_controlled_{target_objective}",
                "efficiency_gap_abs": eff_gap,
                "target_gap_abs": abs(target_gap),
                "target_gap": target_gap,
            })
        if len(pairs) >= n:
            break
    return pairs


def mine_reversal_pairs(
    records: List[Dict],
    preference_templates: Dict[str, List[float]],
    n: int,
    min_margin: float,
) -> List[Dict]:
    templates: List[Tuple[str, List[float]]] = list(preference_templates.items())
    pairs = []
    for a, b in combinations(records, 2):
        for idx, (name_1, w_1) in enumerate(templates):
            margin_1 = _utility(a, w_1) - _utility(b, w_1)
            if abs(margin_1) < min_margin:
                continue
            label_1 = _label(margin_1, 0.0)
            for name_2, w_2 in templates[idx + 1:]:
                margin_2 = _utility(a, w_2) - _utility(b, w_2)
                if abs(margin_2) < min_margin:
                    continue
                label_2 = _label(margin_2, 0.0)
                if label_1 == label_2:
                    continue
                pairs.append({
                    "a_id": a["sample_id"],
                    "b_id": b["sample_id"],
                    "w_1_name": name_1,
                    "w_1": w_1,
                    "label_1": label_1,
                    "margin_1": margin_1,
                    "w_2_name": name_2,
                    "w_2": w_2,
                    "label_2": label_2,
                    "margin_2": margin_2,
                    "sampling_strategy": "reversal",
                })
                if len(pairs) >= n:
                    return pairs
    return pairs
