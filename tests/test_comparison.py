"""Tests for ``behav_utils.analysis.comparison`` — the phase-comparison surface.

A *phase* is a list of ``SessionData`` (the output of ``filter_trials``), pooled
into one trial set. The public entry points are:

    compute_delta_stat   — N phases vs a reference, with permutation p + bootstrap CI
    compute_interaction  — difference of differences between two results

built on the pooling / resampling helpers ``pool_phase_arrays``,
``compute_stats``, ``bootstrap_phase_stats``,
``permute_phase_difference`` and ``summarise_draw_distribution``.

Performance note: the per-permutation / per-bootstrap psychometric fit costs
~0.4 s each and ``compute_delta_stat``'s psychometric *display* object adds a
large fixed CI-band cost, so exercising that path hundreds of times would make
the suite unusably slow (flagged as a real perf smell, not a test artefact).
The delta / interaction machinery is stat-agnostic, so it is driven here with
the fit-free ``accuracy`` statistic; the ``mu`` / ``sigma`` behaviour that needs
a psychometric fit is checked directly via ``compute_stats``, which
fits once.
"""
from datetime import date, timedelta

import numpy as np
import pytest

from behav_utils.analysis.comparison import DeltaStats, Interaction, compute_delta_stat, compute_interaction
from behav_utils.analysis.resampling import bootstrap_phase_stats, permute_phase_difference, summarise_draws
from behav_utils.data.arrays import TrialArrays
from behav_utils.data.structures import SessionData, SessionMetadata, TrialData
from behav_utils.stats import PSYCHOMETRIC, compute_stats


# ── synthetic phase builder (criterion shift = mu displacement) ──────────────
def _phase(n_sessions=2, n=250, shift=0.0, noise=0.15, seed=0, with_aborts=False):
    sessions = []
    for i in range(n_sessions):
        rng = np.random.default_rng(seed * 100 + i)
        stim = rng.uniform(-1, 1, n)
        cat = (stim > 0).astype(float)
        ch = ((stim + rng.normal(0, noise, n) + shift) > 0).astype(float)
        abort = np.zeros(n, bool)
        if with_aborts:                      # ~10% no-response (NaN choice) trials
            miss = rng.choice(n, n // 10, replace=False)
            ch[miss] = np.nan
            abort[miss] = True
        tr = TrialData(trial_number=np.arange(n), stimulus=stim, category=cat,
                       choice=ch, outcome=(ch == cat).astype(float),
                       correct=(ch == cat), abort=abort, opto_on=np.zeros(n, bool))
        sessions.append(SessionData(
            session_id=f's{seed}_{i}', session_idx=i,
            date=date(2026, 1, 1) + timedelta(days=i),
            metadata=SessionMetadata(fields={'stage': 'Full_Task_Cont',
                                             'distribution': 'Uniform'}),
            trials=tr, session_type='regular'))
    return sessions


# fit-free accuracy runs are cheap, so counts can be realistic
FF = dict(n_permutations=60, n_bootstrap=40, seed=1)
SIG = dict(n_permutations=200, n_bootstrap=40, seed=1)


# ── compute_delta_stat (driven with the fit-free accuracy stat) ──────────────
class TestComputeDeltaStat:
    def _r(self, **kw):
        return compute_delta_stat({'ref': _phase(seed=1), 'x': _phase(shift=0.3, seed=2)},
                                  ['accuracy'], reference='ref', curve=False, **{**FF, **kw})

    def test_returns_typed_result(self):
        r = self._r()
        assert isinstance(r, DeltaStats)
        assert set(r.phases) == {'ref', 'x'} and 'x_vs_ref' in r.contrasts
        assert r.contrast('x') is r.contrasts['x_vs_ref']

    def test_phase_summary_shape(self):
        e = self._r().phases['x']
        assert e.n_sessions == 2 and e.n_trials > 0
        assert list(e.stats.index) == ['accuracy']
        assert e.draws['trials'].shape == (FF['n_bootstrap'], 1)
        ci = e.ci('trials').loc['accuracy']
        assert ci['ci_lo'] <= ci['ci_hi']

    def test_contrast_diff_equals_stat_difference(self):
        r = self._r()
        con = r.contrast('x')
        expected = r.phases['x'].stats['accuracy'] - r.phases['ref'].stats['accuracy']
        assert con.diff['accuracy'] == pytest.approx(expected, abs=1e-9)
        assert con.label_a == 'x' and con.label_b == 'ref'
        assert con.n_a > 0 and con.n_b > 0

    def test_table_columns(self):
        t = self._r().table()
        assert {'stat', 'diff', 'ci_lo', 'ci_hi', 'boot_p', 'perm_p', 'unit', 'contrast'} <= set(t.columns)
        assert t['unit'].iloc[0] == 'trials'

    def test_perm_p_in_unit_interval(self):
        assert 0.0 <= self._r().contrast('x').perm_p['accuracy'] <= 1.0

    def test_real_shift_is_flagged(self):
        r = compute_delta_stat({'ref': _phase(seed=1, noise=0.1),
                                'x': _phase(shift=0.6, seed=2, noise=0.1)},
                               ['accuracy'], reference='ref', curve=False, **SIG)
        con = r.contrast('x')
        assert abs(con.diff['accuracy']) > 0.03
        assert con.perm_p['accuracy'] < 0.05

    def test_identical_phases_null_result(self):
        base = _phase(seed=7, noise=0.1)
        r = compute_delta_stat({'ref': base, 'x': base}, ['accuracy'], reference='ref',
                               curve=False, **SIG)
        con = r.contrast('x')
        assert abs(con.diff['accuracy']) < 1e-9
        assert con.perm_p['accuracy'] > 0.05

    def test_list_input_with_labels_matches_dict(self):
        r = compute_delta_stat([_phase(seed=1), _phase(shift=0.3, seed=2)], ['accuracy'],
                               labels=['ref', 'x'], reference='ref', curve=False, **FF)
        assert 'x_vs_ref' in r.contrasts

    def test_three_phases_give_two_contrasts(self):
        r = compute_delta_stat({'ref': _phase(seed=1), 'a': _phase(shift=0.3, seed=2),
                                'b': _phase(shift=-0.3, seed=3)},
                               ['accuracy'], reference='ref', curve=False, **FF)
        assert set(r.contrasts) == {'a_vs_ref', 'b_vs_ref'}

    def test_two_units(self):
        r = self._r(units=('trials', 'sessions'))
        assert r.units == ('trials', 'sessions')
        con = r.contrast('x')
        assert set(con.difference_draws) == {'trials', 'sessions'}
        assert con.boot('sessions').loc['accuracy', 'n_draws'] == FF['n_bootstrap']

    def test_curve_and_update_matrix_readouts(self):
        r = compute_delta_stat({'ref': _phase(seed=1), 'x': _phase(shift=0.3, seed=2)},
                               ['accuracy'], reference='ref', curve=True, update_matrix=True,
                               n_bootstrap=0, n_permutations=0)
        assert r.phases['x'].curve.success and r.phases['x'].update_matrix.n_bins == 8
        assert r.contrast('x').um_diff.shape == (8, 8)

    def test_order_dependent_stat_refused_under_trial_resampling(self):
        with pytest.raises(ValueError):
            compute_delta_stat({'ref': _phase(seed=1), 'x': _phase(seed=2)}, ['w_stimulus'],
                               reference='ref', curve=False, **FF)

    def test_fewer_than_two_phases_raises(self):
        with pytest.raises(ValueError):
            compute_delta_stat({'only': _phase()}, ['accuracy'], reference='only', **FF)

    def test_reference_not_in_phases_raises(self):
        with pytest.raises(ValueError):
            compute_delta_stat({'ref': _phase(seed=1), 'x': _phase(seed=2)}, ['accuracy'],
                               reference='absent', **FF)


class TestComputeInteraction:
    def _paired_results(self):
        C = _phase(seed=3, noise=0.1)
        A = _phase(shift=0.3, seed=1, noise=0.1)
        B = _phase(shift=-0.3, seed=2, noise=0.1)
        rA = compute_delta_stat({'C': C, 'A': A}, ['accuracy'], reference='C', curve=False, **FF)
        rB = compute_delta_stat({'C': C, 'B': B}, ['accuracy'], reference='C', curve=False, **FF)
        return rA, rB

    def test_returns_typed_result(self):
        rA, rB = self._paired_results()
        ix = compute_interaction(rA, rB, 'A_vs_C', contrast_b='B_vs_C')
        assert isinstance(ix, Interaction)
        t = ix.table()
        assert {'delta_a', 'delta_b', 'interaction', 'ci_lo', 'ci_hi', 'p'} <= set(t.columns)
        assert list(t['stat']) == ['accuracy']

    def test_shared_reference_cancellation(self):
        rA, rB = self._paired_results()
        ix = compute_interaction(rA, rB, 'A_vs_C', contrast_b='B_vs_C')
        d_a, d_b = rA.contrast('A').diff['accuracy'], rB.contrast('B').diff['accuracy']
        assert ix.interaction['accuracy'] == pytest.approx(d_a - d_b, abs=1e-9)
        assert ix.delta_a['accuracy'] == pytest.approx(d_a, abs=1e-9)
        assert ix.delta_b['accuracy'] == pytest.approx(d_b, abs=1e-9)

    def test_missing_contrast_raises(self):
        rA, rB = self._paired_results()
        with pytest.raises(KeyError):
            compute_interaction(rA, rB, 'nope')


class TestPsychometricShift:
    def test_criterion_shift_moves_mu_preserves_sigma(self):
        ref = compute_stats(TrialArrays.from_sessions(_phase(seed=1, noise=0.1)), PSYCHOMETRIC)
        shifted = compute_stats(TrialArrays.from_sessions(_phase(shift=0.5, seed=2, noise=0.1)), PSYCHOMETRIC)
        assert abs(shifted['mu'] - ref['mu']) > 0.2
        assert abs(shifted['sigma'] - ref['sigma']) < 0.06


class TestBootstrapPhaseStats:
    def test_shape_and_reproducible(self):
        ph = _phase(seed=1)
        a = bootstrap_phase_stats(ph, ['accuracy'], n_draws=64, seed=0)
        b = bootstrap_phase_stats(ph, ['accuracy'], n_draws=64, seed=0)
        assert a.shape == (64, 1) and list(a.columns) == ['accuracy']
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())

    def test_draws_centre_near_observed(self):
        ph = _phase(seed=1, noise=0.1)
        obs = compute_stats(TrialArrays.from_sessions(ph), ['accuracy'])['accuracy']
        draws = bootstrap_phase_stats(ph, ['accuracy'], n_draws=200, seed=0)['accuracy']
        assert abs(draws.mean() - obs) < 0.05

    def test_session_unit_needs_no_exchangeability(self):
        d = bootstrap_phase_stats(_phase(seed=1, n_sessions=3), ['w_stimulus'], n_draws=10,
                                  seed=0, unit='sessions')
        assert d.shape == (10, 1)
        with pytest.raises(ValueError):
            bootstrap_phase_stats(_phase(seed=1), ['w_stimulus'], n_draws=10, seed=0)


class TestPermutePhaseDifference:
    def test_shape(self):
        d = permute_phase_difference(_phase(seed=1), _phase(shift=0.3, seed=2), ['accuracy'],
                                     n_draws=64, seed=0)
        assert d.shape == (64, 1)

    def test_null_centres_near_zero_for_like_phases(self):
        d = permute_phase_difference(_phase(seed=1, noise=0.1), _phase(seed=1, noise=0.1),
                                     ['accuracy'], n_draws=200, seed=0)['accuracy']
        assert abs(d.mean()) < 0.03


class TestSummariseDraws:
    def test_fields_and_ordering(self):
        draws = np.random.default_rng(0).normal(0.5, 0.1, 500)
        s = summarise_draws(draws, observed=0.5)
        assert s.ci_lo <= s.median <= s.ci_hi
        assert 0.0 <= s.p <= 1.0 and s.n_draws == 500 and s.ci == (s.ci_lo, s.ci_hi)

    def test_far_null_gives_small_p(self):
        draws = np.random.default_rng(0).normal(1.0, 0.05, 500)
        assert summarise_draws(draws, null_value=0.0).p < 0.05

    def test_null_inside_distribution_gives_large_p(self):
        draws = np.random.default_rng(0).normal(0.0, 0.2, 500)
        assert summarise_draws(draws, null_value=0.0).p > 0.2

    def test_too_few_draws_is_nan(self):
        s = summarise_draws([0.1, 0.2, np.nan])
        assert np.isnan(s.p) and s.n_draws == 2
