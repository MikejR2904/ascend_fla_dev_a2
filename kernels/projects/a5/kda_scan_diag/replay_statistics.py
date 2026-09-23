"""Statistics for repeated output hashes, independent of any NPU backend."""
from collections import Counter
import math


def upper_binomial_bound(events: int, trials: int, confidence: float = 0.95) -> float:
    """One-sided exact Clopper–Pearson upper bound, conditional on iid trials.

    Timing-sensitive launches may be correlated; this statistical model is a
    reported assumption, not a proof that an unobserved race cannot occur.
    """
    if (isinstance(events, bool) or isinstance(trials, bool)
            or not isinstance(events, int) or not isinstance(trials, int)
            or trials < 1 or not 0 <= events <= trials
            or not 0 < confidence < 1):
        raise ValueError('Require integer 0 <= events <= trials, trials > 0, 0 < confidence < 1')
    if events == trials:
        return 1.0
    alpha = 1.0 - confidence
    if events == 0:
        return -math.expm1(math.log(alpha) / trials)
    # P_p(X <= events) decreases with p. Log-sum-exp avoids overflowing nCk.
    coefficients = [math.lgamma(trials + 1) - math.lgamma(k + 1)
                    - math.lgamma(trials - k + 1) for k in range(events + 1)]
    lo, hi = 0.0, 1.0
    for _ in range(80):
        p = (lo + hi) / 2
        if p == hi or p == lo:
            break
        terms = [v + k * math.log(p) + (trials - k) * math.log1p(-p)
                 for k, v in enumerate(coefficients)]
        largest = max(terms)
        log_cdf = largest + math.log(sum(math.exp(v-largest) for v in terms))
        if log_cdf > math.log(alpha):
            lo = p
        else:
            hi = p
    return hi


def summarize_hashes(anchor: dict[str, str], samples: list[dict[str, str]]) -> dict:
    """Compare trials to a separately captured anchor, never count it as a trial."""
    if not anchor or not samples or any(set(row) != set(anchor) for row in samples):
        raise ValueError('Nonempty anchor and samples with identical output names required')
    outputs = {}
    for name, expected in sorted(anchor.items()):
        values = [row[name] for row in samples]
        events = sum(value != expected for value in values)
        outputs[name] = dict(anchor_sha256=expected, trials=len(values),
                             hash_distribution=dict(sorted(Counter(values).items())),
                             deviations_from_anchor=events,
                             one_sided_95pct_upper_bound=upper_binomial_bound(events, len(values)),
                             anchor_excluded_from_trials=True)
    return dict(outputs=outputs, confidence_model='conditional independent identically distributed trials',
                scope='Byte-repeatability only; not numerical correctness, race absence, or a repair claim.')
