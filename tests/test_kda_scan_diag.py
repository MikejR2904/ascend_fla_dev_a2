"""Host tests for diagnostic bookkeeping; no kernel qualification implied."""
import importlib.util
import math
from pathlib import Path

import pytest


@pytest.fixture(scope='module')
def stats():
    path = Path(__file__).resolve().parents[1] / 'kernels/projects/a5/kda_scan_diag/replay_statistics.py'
    spec = importlib.util.spec_from_file_location('_kda_scan_statistics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upper_bound_exact_binomial_coverage(stats):
    # Independent small-n polynomial, not the implementation's log-CDF.
    for n, k in ((2, 1), (5, 2), (10, 9), (50, 3)):
        p = stats.upper_binomial_bound(k, n)
        cdf = sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(k+1))
        assert cdf == pytest.approx(.05, abs=2e-13)
    assert stats.upper_binomial_bound(0, 50) == pytest.approx(0.058155079116972264)
    assert stats.upper_binomial_bound(50, 50) == 1


def test_anchor_is_not_a_success_trial(stats):
    result = stats.summarize_hashes({'dh0': 'old', 'dv': 'same'},
                                  [{'dh0': 'new', 'dv': 'same'} for _ in range(50)])
    h0 = result['outputs']['dh0']
    assert h0['trials'] == h0['deviations_from_anchor'] == 50
    assert h0['hash_distribution'] == {'new': 50}
    assert h0['one_sided_95pct_upper_bound'] == 1
    assert result['outputs']['dv']['deviations_from_anchor'] == 0


@pytest.mark.parametrize('events,trials', [(-1,50),(51,50),(0,0),(1.5,50),(True,50)])
def test_reject_invalid_counts(stats, events, trials):
    with pytest.raises(ValueError):
        stats.upper_binomial_bound(events, trials)


def test_missing_output_is_not_silent_success(stats):
    with pytest.raises(ValueError):
        stats.summarize_hashes({'dh0':'a', 'dv':'b'}, [{'dv':'b'}])


@pytest.fixture(scope='module')
def evidence():
    path = Path(__file__).resolve().parents[1] / 'kernels/projects/a5/kda_scan_diag/evidence_tools.py'
    spec = importlib.util.spec_from_file_location('_kda_scan_evidence', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_byte_hash_detects_one_bit_before_conversion(evidence):
    assert evidence.byte_digest(b'abc') == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    assert evidence.byte_digest(bytes([0,128])) != evidence.byte_digest(bytes([0,0]))


def test_redaction_preserves_numeric_fields_environment_lines_and_hashes(evidence):
    private = '/'+'home'+'/'+'example'+'/'
    record = {'trace':private+'task/source.py:358', 'n':50, 'error':.36, 'passed':False,
              'environment':['Version=9.1.0-beta.1','timestamp=20260509_173000235'],
              'sha256':'a'*64, 'absent':None}
    got = evidence.redact_locations(record, {private:'<private-root>/'})
    assert got == {**record, 'trace':'<private-root>/task/source.py:358'}
    assert record['trace'].startswith(private)


def test_unmapped_location_and_key_collision_are_rejected(evidence):
    private = '/'+'home'+'/'+'example'+'/'
    with pytest.raises(ValueError, match='Private home'):
        evidence.redact_locations({'trace':private+'task'}, {})
    address = '.'.join(map(str,(192,0,2,7)))
    with pytest.raises(ValueError, match='Machine address'):
        evidence.require_shareable({'endpoint':address})
    with pytest.raises(ValueError, match='merge'):
        evidence.redact_locations({'private-a':1,'private-b':2}, {'private-a':'x','private-b':'x'})
