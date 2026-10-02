"""Tests for behav_utils.data.selection.SessionFilter."""

from datetime import date

import numpy as np

from behav_utils.data.ops.selection import SessionFilter, list_presets

from .conftest import _make_session, _make_trial_data


class TestSessionFilterBasic:
    """Basic filtering tests."""

    def test_no_constraints(self, synthetic_animal):
        """Empty filter returns all sessions."""
        f = SessionFilter()
        result = f.apply(synthetic_animal)
        assert len(result) == len(synthetic_animal.sessions)

    def test_stage_filter(self, synthetic_animal):
        """Filter by stage name."""
        f = SessionFilter(stage='Full_Task_Cont')
        result = f.apply(synthetic_animal)
        assert len(result) == len(synthetic_animal.sessions)

        f2 = SessionFilter(stage='Nonexistent')
        result2 = f2.apply(synthetic_animal)
        assert len(result2) == 0

    def test_distribution_filter(self, synthetic_animal):
        """Filter by distribution."""
        f = SessionFilter(distribution='Uniform')
        result = f.apply(synthetic_animal)
        assert len(result) == len(synthetic_animal.sessions)

    def test_min_trials(self, synthetic_animal):
        """Filter by minimum trial count."""
        f = SessionFilter(min_trials=500)
        result = f.apply(synthetic_animal)
        # Fixture has 300 trials per session
        assert len(result) == 0

        f2 = SessionFilter(min_trials=100)
        result2 = f2.apply(synthetic_animal)
        assert len(result2) == len(synthetic_animal.sessions)


class TestExcludeTypes:
    """exclude_types: the library excludes nothing by default; a preset states what it drops."""

    def test_default_excludes_nothing(self, synthetic_animal):
        f = SessionFilter()
        assert f.exclude_types == ()
        assert len(f.apply(synthetic_animal)) == len(synthetic_animal.sessions)

    def test_exclude_masking(self, synthetic_animal):
        f = SessionFilter(exclude_types=('masking',))
        result = f.apply(synthetic_animal)
        assert all(s.session_type != 'masking' for s in result)
        masking_count = sum(1 for s in synthetic_animal.sessions if s.session_type == 'masking')
        assert len(result) == len(synthetic_animal.sessions) - masking_count

    def test_describe_mentions_types(self):
        assert 'masking' in SessionFilter(exclude_types=('masking', 'washout')).describe()


class TestPresets:
    """Tests for preset registry."""

    def test_presets_exist(self):
        """At least some presets should be registered."""
        presets = list_presets()
        assert len(presets) > 0
        assert 'expert_uniform' in presets

    def test_preset_returns_filter(self):
        """Each preset should be a SessionFilter."""
        presets = list_presets()
        for name in presets:
            from behav_utils.data.ops.selection import get_preset
            f = get_preset(name)
            assert isinstance(f, SessionFilter)


class TestWithOverrides:
    """Tests for SessionFilter.with_overrides()."""

    def test_override_creates_new(self):
        """with_overrides returns a new object."""
        base = SessionFilter(min_trials=100)
        modified = base.with_overrides(min_trials=200)
        assert base.min_trials == 100
        assert modified.min_trials == 200
        assert base is not modified


class TestSessionType:
    """Tests for session_type filtering."""

    def test_select_opto_sessions(self, synthetic_opto_animal):
        """session_type='opto' returns only opto sessions."""
        f = SessionFilter(session_type='opto')
        result = f.apply(synthetic_opto_animal)
        # Sessions 7-11 (Uniform opto) + 14-18 (Asym_Right opto) = 10
        assert len(result) == 10
        for sess in result:
            assert np.any(sess.trials.opto_on)

    def test_select_masking_sessions(self, synthetic_opto_animal):
        """session_type='masking' returns only masking sessions."""
        f = SessionFilter(session_type='masking')
        result = f.apply(synthetic_opto_animal)
        # Sessions 5-6 = 2
        assert len(result) == 2
        for sess in result:
            assert sess.session_type == 'masking'

    def test_select_washout_sessions(self, synthetic_opto_animal):
        """session_type='washout' returns only washout sessions."""
        f = SessionFilter(session_type='washout')
        result = f.apply(synthetic_opto_animal)
        # Sessions 12-13 = 2
        assert len(result) == 2
        for sess in result:
            assert sess.session_type == 'washout'

    def test_select_regular_sessions(self, synthetic_opto_animal):
        """session_type='regular' returns only regular sessions."""
        f = SessionFilter(session_type='regular')
        result = f.apply(synthetic_opto_animal)
        # Sessions 0-4 (baseline) + 19-23 (recovery) = 10
        assert len(result) == 10
        for sess in result:
            assert not getattr(sess, 'masking', False)
            assert not getattr(sess, 'washout', False)
            assert not np.any(sess.trials.opto_on)

    def test_session_type_list(self, synthetic_opto_animal):
        """session_type as list selects multiple types."""
        f = SessionFilter(session_type=['opto', 'masking'])
        result = f.apply(synthetic_opto_animal)
        # 10 opto + 2 masking = 12
        assert len(result) == 12

    def test_session_type_overrides_exclude_types(self, synthetic_opto_animal):
        """When session_type is set, exclude_types is ignored."""
        f = SessionFilter(session_type='masking', exclude_types=('masking',))
        result = f.apply(synthetic_opto_animal)
        assert len(result) == 2  # still gets masking sessions

    def test_session_type_with_distribution(self, synthetic_opto_animal):
        """Combine session_type with distribution filter."""
        f = SessionFilter(distribution='Uniform', session_type='opto')
        result = f.apply(synthetic_opto_animal)
        # Sessions 7-11 = 5 (Uniform opto only, not Asym_Right opto)
        assert len(result) == 5


class TestWashoutExclusion:
    """washout is just another project session type."""

    def test_washout_excluded_when_asked(self, synthetic_opto_animal):
        result = SessionFilter(exclude_types=('washout',)).apply(synthetic_opto_animal)
        assert all(s.session_type != 'washout' for s in result)

    def test_washout_included_by_default(self, synthetic_opto_animal):
        result = SessionFilter().apply(synthetic_opto_animal)
        assert sum(1 for s in result if s.session_type == 'washout') == 2


class TestListDistribution:
    """Tests for list-valued distribution filters."""

    def test_list_distribution(self, synthetic_opto_animal):
        """Distribution as list matches any."""
        f = SessionFilter(
            distribution=['Uniform', 'Asym_Right'],
                    )
        result = f.apply(synthetic_opto_animal)
        assert len(result) == len(synthetic_opto_animal.sessions)

    def test_list_distribution_subset(self, synthetic_opto_animal):
        """List distribution with session_type narrows correctly."""
        f = SessionFilter(
            distribution=['Uniform', 'Asym_Right'],
            session_type='opto',
        )
        result = f.apply(synthetic_opto_animal)
        # All opto sessions (both distributions) = 10
        assert len(result) == 10


class TestSessionTable:
    """Tests for session_table washout classification."""

    def test_session_table_washout_type(self, synthetic_opto_animal):
        """session_table classifies washout sessions correctly."""
        table = synthetic_opto_animal.session_table
        washout_rows = table[table['session_type'] == 'washout']
        assert len(washout_rows) == 2

    def test_session_table_four_types(self, synthetic_opto_animal):
        """session_table produces all four session types."""
        table = synthetic_opto_animal.session_table
        types = set(table['session_type'].unique())
        assert types == {'regular', 'masking', 'opto', 'washout'}


class TestResolveSessionType:
    """Tests for _resolve_session_type priority."""

    def test_washout_takes_priority(self, rng):
        """A session marked as both washout and masking is 'washout'."""
        trials = _make_trial_data(100, rng, opto_frac=0.0)
        sess = _make_session(0, date(2026, 1, 1), trials,
                             masking=True, washout=True)
        assert SessionFilter._resolve_session_type(sess) == 'washout'


class TestBlocksAndSwitches:
    def _seq(self, spec):
        from types import SimpleNamespace as N
        return [N(distribution=d, session_idx=i, trials=N(choice=np.zeros(n))) for i, (d, n) in enumerate(spec)]

    def test_blocks_and_switches(self):
        from behav_utils.data import block_after, block_before, find_blocks, find_switches
        seq = self._seq([('U', 3000), ('U', 3000), ('B', 2000), ('B', 2500), ('A', 3700)])
        blocks = find_blocks(seq)
        assert [(b.distribution, b.n_sessions, b.n_trials) for b in blocks] == [('U', 2, 6000), ('B', 2, 4500), ('A', 1, 3700)]
        sw = find_switches(seq)
        assert [(s['from_distribution'], s['to_distribution'], s['trial_index']) for s in sw] == [('U', 'B', 6000), ('B', 'A', 10500)]
        assert [s.session_idx for s in block_after(seq, sw[0])] == [2, 3]
        assert [s.session_idx for s in block_before(seq, sw[0])] == [0, 1]

    def test_short_excursion_is_absorbed(self):
        from behav_utils.data import find_blocks, find_switches
        seq = self._seq([('U', 8000), ('B', 600), ('U', 300), ('B', 2000), ('B', 2500), ('A', 3700)])
        assert len(find_switches(seq)) == 4                                # raw: every change counts
        blocks = find_blocks(seq, min_block_trials=1000)
        assert [(b.distribution, b.n_trials) for b in blocks] == [('U', 8900), ('B', 4500), ('A', 3700)]
        sw = find_switches(seq, min_block_trials=1000)
        assert [(s['from_distribution'], s['to_distribution']) for s in sw] == [('U', 'B'), ('B', 'A')]
        assert sw[0]['n_trials_after'] == 4500 and sw[0]['n_sessions_after'] == 2
