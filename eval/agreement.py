"""STEP 4 — rules-vs-LLM agreement maths + the final-grade reconciliation policy.

Two independent graders only become trustworthy ground truth if we MEASURE how
much they agree (and surface where they don't). Pure stdlib — scipy is not a
project dependency — so these are implemented and unit-tested directly.
"""

from __future__ import annotations

import math


def _rankdata(values):
    """Average-rank of each value (ties share the mean of their rank span)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x, y):
    n = len(x)
    if n == 0:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def spearman(x, y):
    """Spearman rank correlation = Pearson on average ranks."""
    if len(x) < 2:
        return 0.0
    return _pearson(_rankdata(x), _rankdata(y))


def kendall_tau(x, y):
    """Kendall tau-b (handles ties in either variable)."""
    n = len(x)
    if n < 2:
        return 0.0
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            s = (dx > 0) - (dx < 0)
            t = (dy > 0) - (dy < 0)
            if s == 0 and t == 0:
                continue
            if s == 0:
                tx += 1
            elif t == 0:
                ty += 1
            elif s * t > 0:
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    if denom == 0:
        return 0.0
    return (conc - disc) / denom


def agreement_report(pairs):
    """pairs: list of (grade_rules, grade_llm) with both present (ints)."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "exact_match": None, "within_1": None, "spearman": None, "kendall": None}
    xr = [a for a, _ in pairs]
    xl = [b for _, b in pairs]
    exact = sum(1 for a, b in pairs if a == b) / n
    within1 = sum(1 for a, b in pairs if abs(a - b) <= 1) / n
    return {
        "n": n,
        "exact_match": round(exact, 4),
        "within_1": round(within1, 4),
        "spearman": round(spearman(xr, xl), 4),
        "kendall": round(kendall_tau(xr, xl), 4),
    }


def reconcile(grade_rules_val, grade_llm_val, forced_zero):
    """Final-grade policy (STEP 4):

    - honeypot/stuffer -> 0 (rules win; objective).
    - LLM grade absent -> rule-only fallback (final = grade_rules).
    - agree within 1 -> rounded average.
    - disagree by >= 2 -> final = None, flag for human review.
    """
    if forced_zero:
        return 0, False
    if grade_llm_val is None:
        return grade_rules_val, False
    if abs(grade_rules_val - grade_llm_val) >= 2:
        return None, True
    return int(math.floor((grade_rules_val + grade_llm_val) / 2.0 + 0.5)), False
