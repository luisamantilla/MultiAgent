from __future__ import annotations
from typing import Dict, List, Tuple, Set, Any, Optional
import json
import re
from math import inf

# =====================
# Lexical evaluator (v1)
# =====================

# -------- Canonicalization helpers ---------
_word_re = re.compile(r"[A-Za-z0-9\+]+")

def canon(s: str) -> str:
    s = s.lower()
    s = s.replace("γ", "g").replace("β", "b").replace("α", "a")
    s = s.replace("ifn-γ", "ifng").replace("ifn-α", "ifna").replace("ifn-β", "ifnb")
    s = s.replace("+", "plus")
    tokens = _word_re.findall(s)
    return "_".join(tokens)


def token_set(s: str) -> Set[str]:
    return set(t for t in _word_re.findall(s.lower()) if len(t) > 2)


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# -------- Ground truth loading & extraction ---------

def load_ground_truth(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def extract_truth_sets(gt: Dict[str, Any], truth_mode: str = "auto") -> Dict[str, Set[str]]:
    cells = {canon(x) for x in gt.get("cell_types", [])}
    mols = {canon(x) for x in gt.get("molecules", [])}
    # include other_entities as cells/entities
    for x in gt.get("other_entities", []):
        cells.add(canon(x))

    # Interactions: control extraction via truth_mode
    inters: Set[str] = set()
    explicit = gt.get("interactions")
    if truth_mode == "explicit":
        if isinstance(explicit, list):
            for x in explicit:
                if isinstance(x, str):
                    inters.add(canon(x))
    elif truth_mode == "names":
        for proc in gt.get("BiologicalProcesses", []):
            name = proc.get("name")
            if isinstance(name, str):
                inters.add(canon(name))
    elif truth_mode == "steps":
        # Count both string lines and dict steps (concatenate step + detail)
        for proc in gt.get("BiologicalProcesses", []):
            for step in proc.get("steps", []):
                if isinstance(step, str):
                    inters.add(canon(step))
                elif isinstance(step, dict):
                    s = f"{step.get('step','')} {step.get('detail','')}".strip()
                    if s:
                        inters.add(canon(s))
    else:  # auto: prefer explicit > names+steps
        if isinstance(explicit, list) and explicit:
            for x in explicit:
                if isinstance(x, str):
                    inters.add(canon(x))
        else:
            for proc in gt.get("BiologicalProcesses", []):
                name = proc.get("name")
                if isinstance(name, str):
                    inters.add(canon(name))
                for step in proc.get("steps", []):
                    if isinstance(step, str):
                        inters.add(canon(step))
                    elif isinstance(step, dict):
                        s = f"{step.get('step','')} {step.get('detail','')}"
                        inters.add(canon(s))
    return {"cell_types": cells, "molecules": mols, "interactions": inters}


# -------- Prediction extraction ---------

def extract_pred_sets(result: Dict[str, Any]) -> Dict[str, Set[str]]:
    pi = result.get("pi", {})
    sel = pi.get("selected", {})
    cells = {canon(x) for x in sel.get("cell_types", [])}
    mols = {canon(x) for x in sel.get("molecules", [])}
    inters = {canon(x) for x in sel.get("interactions", [])}
    return {"cell_types": cells, "molecules": mols, "interactions": inters}


# -------- Fuzzy matching for interactions ---------

def fuzzy_match(pred: Set[str], truth: Set[str], threshold: float = 0.45) -> Tuple[int, int, int, List[Tuple[str, str, float]]]:
    matches: List[Tuple[str, str, float]] = []
    used_truth: Set[str] = set()
    tp = 0
    for p in pred:
        p_tokens = set(p.split("_")) if p else set()
        best_t = None
        best_sim = 0.0
        for t in truth:
            if t in used_truth:
                continue
            sim = jaccard(p_tokens, set(t.split("_")))
            if sim > best_sim:
                best_sim = sim
                best_t = t
        if best_t is not None and best_sim >= threshold:
            tp += 1
            used_truth.add(best_t)
            matches.append((p, best_t, best_sim))
    fp = max(0, len(pred) - tp)
    fn = max(0, len(truth) - tp)
    return tp, fp, fn, matches


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


# -------- Evaluation ---------

def evaluate(result: Dict[str, Any], ground_truth: Dict[str, Any], max_interactions: int | None = None, truth_mode: str = "auto") -> Dict[str, Any]:
    truth_sets = extract_truth_sets(ground_truth, truth_mode=truth_mode)
    pred_sets = extract_pred_sets(result)

    # Cells and molecules: exact canonical match
    tp_c = len(pred_sets["cell_types"] & truth_sets["cell_types"])
    fp_c = len(pred_sets["cell_types"] - truth_sets["cell_types"])
    fn_c = len(truth_sets["cell_types"] - pred_sets["cell_types"])

    tp_m = len(pred_sets["molecules"] & truth_sets["molecules"])
    fp_m = len(pred_sets["molecules"] - truth_sets["molecules"])
    fn_m = len(truth_sets["molecules"] - pred_sets["molecules"])

    # Interactions: fuzzy
    tp_i, fp_i, fn_i, matches = fuzzy_match(pred_sets["interactions"], truth_sets["interactions"], threshold=0.45)

    metrics = {
        "cell_types": prf(tp_c, fp_c, fn_c),
        "molecules": prf(tp_m, fp_m, fn_m),
        "interactions": prf(tp_i, fp_i, fn_i),
    }

    budget = None
    if max_interactions is not None:
        over = max(0, len(pred_sets["interactions"]) - max_interactions)
        budget = {
            "max_interactions": max_interactions,
            "selected": len(pred_sets["interactions"]),
            "over_budget": over,
            "penalty": over,  # simple linear penalty; can customize
        }

    return {
        "metrics": metrics,
        "counts": {
            "truth": {k: len(v) for k, v in truth_sets.items()},
            "pred": {k: len(v) for k, v in pred_sets.items()},
        },
        "interaction_matches": matches[:25],
        "budget": budget,
    }


# ==========================
# Semantic evaluator (v2)
# ==========================

_WORD2 = re.compile(r"[A-Za-z0-9\+\-\./]+")
_GREEK = {"γ": "g", "β": "b", "α": "a", "Δ": "d", "δ": "d"}
_ALIASES = {
    "ifn-γ": "ifng", "ifnγ": "ifng", "ifn-gamma": "ifng", "ifn gamma": "ifng",
    "ifn-α": "ifna", "ifnα": "ifna", "ifn-alpha": "ifna", "ifn alpha": "ifna",
    "ifn-β": "ifnb", "ifnβ": "ifnb", "ifn-beta": "ifnb", "ifn beta": "ifnb",
    "tnf-α": "tnfa", "tnfα": "tnfa", "tnf-alpha": "tnfa", "tnf alpha": "tnfa",
    "il-1β": "il1b", "il1β": "il1b", "il-1b": "il1b", "il1b": "il1b",
    "cd8+ t": "cd8 t cell", "cd8 t": "cd8 t cell", "cd8+ t cell": "cd8 t cell",
    "treg": "cd4 treg", "regulatory t cell": "cd4 treg",
}

def _canon_text2(s: str) -> str:
    s = s.lower()
    for k, v in _GREEK.items():
        s = s.replace(k, v)
    for k, v in _ALIASES.items():
        s = re.sub(rf"\b{k}\b", v, s)
    s = s.replace("+", " + ")
    toks = _WORD2.findall(s)
    return " ".join(toks)

def _canon_entity(s: str) -> str:
    return _canon_text2(s).replace("  ", " ").strip()

_REL_NORMALIZE = {
    "activate": "activates", "activates": "activates", "stimulate": "activates", "induces": "activates", "promotes": "activates",
    "inhibit": "inhibits", "inhibits": "inhibits", "suppress": "inhibits", "represses": "inhibits", "blocks": "inhibits",
    "secretes": "secretes", "releases": "secretes", "produces": "secretes",
    "recruits": "recruits", "attracts": "recruits", "chemotactic_for": "recruits",
    "differentiates": "differentiates", "drives": "differentiates",
    "presents": "presents", "antigen_presentation": "presents",
    # added noun-phrase normalized relations
    "binds": "binds", "attachment": "binds", "binding": "binds",
    "enters": "enters", "entry": "enters",
    "kills": "kills", "cytotoxic_killing": "kills", "killing": "kills",
    "phagocytoses": "phagocytoses", "phagocytosis": "phagocytoses",
}

_VERB_PAT = r"(activates?|stimulates?|induces?|promotes?|inhibits?|suppresses?|represses?|blocks?|secretes?|releases?|produces?|recruits?|attracts?|presents?|differentiates?|binds?|enters?|kills?|phagocytoses?)"
_VIA_PAT = r"(?:via|through|by means of)\s+([^.;,]+)"

def _normalize_rel(verb: str) -> Optional[str]:
    v = _canon_text2(verb).replace(" ", "")
    for k, vn in _REL_NORMALIZE.items():
        if v.startswith(k):
            return vn
    return None

_HEAD_NP_TO_REL = {
    "secretion": "secretes",
    "activation": "activates",
    "induction": "activates",
    "recruitment": "recruits",
    "phagocytosis": "phagocytoses",
    "killing": "kills",
    "presentation": "presents",
    "entry": "enters",
    "attachment": "binds",
    "binding": "binds",
}

def _parse_via_segment(txt: str) -> List[str]:
    # split on commas/semicolons/and; keep simple tokens
    items = [t.strip() for t in re.split(r"[;,]|\band\b", txt) if t.strip()]
    return [re.sub(r"^[()\s]+|[()\s]+$", "", x) for x in items]

def _extract_nominal_interaction(text: str) -> Optional[Dict[str, Any]]:
    t = text.strip()
    if not t:
        return None
    # Pattern 1: "X recruitment by Y (A, B)"
    m = re.search(r"^(.+?)\s+recruitment\s+by\s+([^()]+)(?:\s*\(([^)]+)\))?", t, flags=re.IGNORECASE)
    if m:
        obj, subj, via = m.group(1).strip(), m.group(2).strip(), m.group(3)
        return {
            "subj": subj,
            "rel": "recruits",
            "obj": obj,
            "polarity": "positive",
            "via": _parse_via_segment(via) if via else [],
            "context": {},
            "evidence": text,
        }
    # Pattern 2: "<Head> of X by Y"
    m = re.search(r"^(secretion|activation|induction|phagocytosis|killing|presentation)\s+of\s+(.+?)\s+by\s+(.+)$", t, flags=re.IGNORECASE)
    if m:
        head, obj, subj = m.group(1).lower(), m.group(2).strip(), m.group(3).strip()
        rel = _HEAD_NP_TO_REL.get(head)
        if rel:
            pol = "positive" if rel in {"activates", "secretes", "recruits", "phagocytoses", "kills", "presents"} else "neutral"
            return {"subj": subj, "rel": rel, "obj": obj, "polarity": pol, "via": [], "context": {}, "evidence": text}
    # Pattern 3: "X attachment/binding to Y" or "X entry into Y"
    m = re.search(r"^(.+?)\s+(attachment|binding)\s+(?:to|onto|with)\s+(.+)$", t, flags=re.IGNORECASE)
    if m:
        subj, head, obj = m.group(1).strip(), m.group(2).lower(), m.group(3).strip()
        rel = _HEAD_NP_TO_REL.get(head, "binds")
        return {"subj": subj, "rel": rel, "obj": obj, "polarity": "positive", "via": [], "context": {}, "evidence": text}
    m = re.search(r"^(.+?)\s+entry\s+into\s+(.+)$", t, flags=re.IGNORECASE)
    if m:
        subj, obj = m.group(1).strip(), m.group(2).strip()
        return {"subj": subj, "rel": "enters", "obj": obj, "polarity": "positive", "via": [], "context": {}, "evidence": text}
    return None

def extract_interactions_raw(lines: List[str]) -> List[Dict[str, Any]]:
    interactions: List[Dict[str, Any]] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        # Try nominal phrase patterns first
        nom = _extract_nominal_interaction(text)
        if nom:
            interactions.append(nom)
            continue
        m = re.search(rf"(.+?)\s+{_VERB_PAT}\s+(.+?)(?:[.;]|$)", text, flags=re.IGNORECASE)
        if not m:
            continue
        subj_raw, verb_raw, obj_raw = m.group(1), m.group(2), m.group(3)
        rel = _normalize_rel(verb_raw)
        if not rel:
            continue
        via: List[str] = []
        vm = re.search(_VIA_PAT, text, flags=re.IGNORECASE)
        if vm:
            via_txt = vm.group(1)
            via = [v.strip(" ,.") for v in re.split(r"[;,]|\band\b", via_txt) if v.strip()]
        interactions.append({
            "subj": subj_raw.strip(),
            "rel": rel,
            "obj": obj_raw.strip(),
            "polarity": "positive" if rel in {"activates", "secretes", "recruits", "differentiates", "presents"} else (
                "negative" if rel in {"inhibits"} else "neutral"
            ),
            "via": via,
            "context": {},
            "evidence": text,
        })
    return interactions

def _entity_sim(a: str, b: str) -> float:
    a, b = _canon_entity(a), _canon_entity(b)
    if a == b:
        return 1.0
    at, bt = set(a.split()), set(b.split())
    if not at or not bt:
        return 0.0
    j = len(at & bt) / len(at | bt)
    head_bonus = 0.1 if (a.split()[0] == b.split()[0]) else 0.0
    return min(1.0, j + head_bonus)

def _list_jaccard(a: List[str], b: List[str]) -> float:
    A = {_canon_entity(x) for x in a}
    B = {_canon_entity(x) for x in b}
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def _pair_score(t: Dict[str, Any], p: Dict[str, Any]) -> float:
    s_subj = _entity_sim(t["subj"], p["subj"]) if t.get("subj") and p.get("subj") else 0.0
    s_obj = _entity_sim(t["obj"], p["obj"]) if t.get("obj") and p.get("obj") else 0.0
    s_ent = 0.5 * s_subj + 0.5 * s_obj
    s_rel = 1.0 if t.get("rel") == p.get("rel") else 0.0
    s_pol = 1.0 if t.get("polarity") == p.get("polarity") else 0.0
    s_via = _list_jaccard(t.get("via", []), p.get("via", []))
    t_ctx, p_ctx = t.get("context", {}), p.get("context", {})
    ctxt_keys = set(t_ctx).intersection(p_ctx)
    s_ctx = 0.0
    if ctxt_keys:
        same = sum(1 for k in ctxt_keys if _canon_entity(t_ctx[k]) == _canon_entity(p_ctx[k]))
        s_ctx = same / len(ctxt_keys)
    w_ent, w_rel, w_pol, w_via, w_ctx = 0.45, 0.35, 0.10, 0.07, 0.03
    return w_ent * s_ent + w_rel * s_rel + w_pol * s_pol + w_via * s_via + w_ctx * s_ctx

def _hungarian_matches(cost: List[List[float]]) -> List[Tuple[int, int]]:
    try:
        import importlib
        np = importlib.import_module("numpy")
        scipy_opt = importlib.import_module("scipy.optimize")
        C = np.array(cost, dtype=float)
        row_ind, col_ind = scipy_opt.linear_sum_assignment(C)
        return [(int(i), int(j)) for i, j in zip(row_ind, col_ind)]
    except Exception:
        # Greedy fallback
        pairs: List[Tuple[int, int]] = []
        used_r, used_c = set(), set()
        flat: List[Tuple[float, int, int]] = []
        for i in range(len(cost)):
            for j in range(len(cost[i])):
                flat.append((cost[i][j], i, j))
        flat.sort()
        for c, i, j in flat:
            if i in used_r or j in used_c:
                continue
            used_r.add(i); used_c.add(j); pairs.append((i, j))
        return pairs

def _gt_lines_from_truth(gt: Dict[str, Any], truth_mode: str = "auto") -> List[str]:
    lines: List[str] = []
    explicit = gt.get("interactions")
    if truth_mode == "explicit":
        if isinstance(explicit, list):
            lines = [str(x) for x in explicit]
    elif truth_mode == "names":
        for proc in gt.get("BiologicalProcesses", []):
            if isinstance(proc.get("name"), str):
                lines.append(proc["name"])
    elif truth_mode == "steps":
        for proc in gt.get("BiologicalProcesses", []):
            for step in proc.get("steps", []):
                if isinstance(step, str):
                    lines.append(step)
                elif isinstance(step, dict):
                    s = f"{step.get('step','')} {step.get('detail','')}".strip()
                    if s:
                        lines.append(s)
    else:  # auto
        if isinstance(explicit, list) and explicit:
            lines = [str(x) for x in explicit]
        else:
            for proc in gt.get("BiologicalProcesses", []):
                if isinstance(proc.get("name"), str):
                    lines.append(proc["name"])
                for step in proc.get("steps", []):
                    if isinstance(step, str):
                        lines.append(step)
                    elif isinstance(step, dict):
                        s = f"{step.get('step','')} {step.get('detail','')}".strip()
                        if s:
                            lines.append(s)
    return lines

def build_interactions_from_gt(gt: Dict[str, Any], truth_mode: str = "auto") -> List[Dict[str, Any]]:
    lines = _gt_lines_from_truth(gt, truth_mode=truth_mode)
    return extract_interactions_raw(lines)

def build_interactions_from_pred(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    lines: List[str] = []
    # PI selected
    sel = result.get("pi", {}).get("selected", {})
    lines += [str(x) for x in sel.get("interactions", [])]
    # Computationalist interactions
    comp = result.get("computationalist", {})
    lines += [str(x) for x in comp.get("interactions", [])]
    # Biologist keeps
    bio = result.get("biologist", {})
    lines += [str(x) for x in bio.get("keep_interactions", [])]
    # Experimentalist keeps
    exp = result.get("experimentalist", {})
    lines += [str(x) for x in exp.get("keep_interactions", [])]
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for s in lines:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return extract_interactions_raw(uniq)

def evaluate_interactions(truth: List[Dict[str, Any]], preds: List[Dict[str, Any]], threshold: float = 0.60) -> Dict[str, Any]:
    # Score matrix (cost = 1 - score)
    cost: List[List[float]] = []
    for t in truth:
        row: List[float] = []
        for p in preds:
            s = _pair_score(t, p)
            row.append(1.0 - s)
        cost.append(row)

    matches: List[Tuple[int, int, float]] = []
    t_used, p_used = set(), set()

    if truth and preds:
        pairs = _hungarian_matches(cost)
        for (i, j) in pairs:
            s = 1.0 - cost[i][j] if i < len(cost) and j < len(cost[i]) else 0.0
            if s >= threshold:
                matches.append((i, j, s))
                t_used.add(i); p_used.add(j)

    tp = len(matches)
    fp = len(preds) - tp
    fn = len(truth) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

    side_by_side = []
    for (i, j, s) in sorted(matches, key=lambda x: -x[2]):
        t, p = truth[i], preds[j]
        side_by_side.append({
            "score": round(s, 3),
            "truth": {
                "subj": t.get("subj"), "rel": t.get("rel"), "obj": t.get("obj"),
                "polarity": t.get("polarity"), "via": t.get("via", []), "evidence": t.get("evidence", "")
            },
            "pred": {
                "subj": p.get("subj"), "rel": p.get("rel"), "obj": p.get("obj"),
                "polarity": p.get("polarity"), "via": p.get("via", []), "evidence": p.get("evidence", "")
            },
            "facet_scores": {
                "entity_sim": round(0.5 * _entity_sim(t.get("subj", ""), p.get("subj", "")) + 0.5 * _entity_sim(t.get("obj", ""), p.get("obj", "")), 3),
                "rel_match": 1.0 if t.get("rel") == p.get("rel") else 0.0,
                "polarity_match": 1.0 if t.get("polarity") == p.get("polarity") else 0.0,
                "via_overlap": round(_list_jaccard(t.get("via", []), p.get("via", [])), 3),
            }
        })

    missed = [truth[i] for i in range(len(truth)) if i not in t_used]
    extra = [preds[j] for j in range(len(preds)) if j not in p_used]

    return {
        "metrics": {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)},
        "counts": {"truth_interactions": len(truth), "pred_interactions": len(preds), "matched": tp},
        "matches": side_by_side,
        "missed_truth": [{"evidence": x.get("evidence", ""), **{k: x.get(k) for k in ("subj", "rel", "obj")}} for x in missed],
        "extra_pred": [{"evidence": x.get("evidence", ""), **{k: x.get(k) for k in ("subj", "rel", "obj")}} for x in extra],
        "threshold": threshold,
        "weights": {"entity": 0.45, "relation": 0.35, "polarity": 0.10, "via": 0.07, "context": 0.03},
    }

def evaluate_v2(result: Dict[str, Any], ground_truth: Dict[str, Any], threshold: float = 0.60, truth_mode: str = "auto") -> Dict[str, Any]:
    truth = build_interactions_from_gt(ground_truth, truth_mode=truth_mode)
    preds = build_interactions_from_pred(result)
    return evaluate_interactions(truth, preds, threshold=threshold)
