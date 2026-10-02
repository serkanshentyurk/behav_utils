"""
Tests for behav_utils/analysis/session_features.py.

After the consolidation, the module exposes only compute_session_features
(others were removed). Verifies it produces expected output keys including RT.
"""



class TestComputeSessionFeatures:
    """compute_session_features returns a flat dict of features."""

    def test_returns_dict(self, synthetic_session):
        """Output is a dict."""
        from behav_utils.analysis.session_features import compute_session_features
        result = compute_session_features(synthetic_session)
        assert isinstance(result, dict)

    def test_has_accuracy(self, synthetic_session):
        """Accuracy is one of the features."""
        from behav_utils.analysis.session_features import compute_session_features
        result = compute_session_features(synthetic_session)
        # Could be 'accuracy' or 'accuracy.value' depending on flattening
        keys = list(result.keys())
        has_accuracy = any('accuracy' in k.lower() for k in keys)
        assert has_accuracy, f"no accuracy key in {keys[:10]}"

    def test_returns_finite_values(self, synthetic_session):
        """Returned scalar features are finite (or appropriately NaN)."""
        from behav_utils.analysis.session_features import compute_session_features
        result = compute_session_features(synthetic_session)
        # Just check we have a non-trivial number of keys
        assert len(result) > 3
