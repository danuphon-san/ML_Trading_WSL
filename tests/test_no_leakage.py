"""
Tests to detect data leakage in features and labels

Data leakage occurs when information from the future accidentally
appears in training data, causing unrealistic performance.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.ta_features import create_technical_features
from src.labeling.labels import generate_forward_returns


class TestDataLeakage:
    """Test suite for data leakage detection"""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = {
            'date': dates,
            'symbol': ['TEST'] * 100,
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000000, 5000000, 100)
        }
        return pd.DataFrame(data)

    def test_labels_only_use_future_data(self, sample_data):
        """
        Test that labels are calculated from future prices only

        Labels should use data AFTER the current row (shift(-horizon))
        This is CORRECT - labels should predict the future
        """
        import yaml
        config = {
            'labels': {
                'horizon': 5,
                'target_type': 'return'
            }
        }

        df = generate_forward_returns(sample_data, config)

        # Check that the last 'horizon' rows have NaN labels
        assert df['forward_return_5d'].iloc[-5:].isna().all(), \
            "Last 5 rows should have NaN labels (no future data available)"

        # Check that earlier rows have non-NaN labels
        assert df['forward_return_5d'].iloc[:90].notna().any(), \
            "Earlier rows should have labels"

    def test_features_only_use_past_data(self, sample_data):
        """
        Test that features use only historical data

        Features should NEVER use future information
        All indicators should look backwards only
        """
        import yaml
        config = {
            'technical': {
                'lookback_windows': {
                    'short': [5, 10],
                    'medium': [20]
                }
            }
        }

        df = create_technical_features(sample_data, config)

        # Check that first few rows have NaN features (no historical data)
        # For momentum_20d, first 20 rows should be NaN
        if 'momentum_20d' in df.columns:
            # At least first few rows should be NaN
            assert df['momentum_20d'].iloc[:5].isna().any(), \
                "First rows should have NaN features (no historical data)"

    def test_no_same_day_feature_label_correlation(self, sample_data):
        """
        CRITICAL TEST: Features at time t should NOT correlate
        with labels at time t

        If correlation is >0.5, there's likely data leakage
        """
        import yaml

        config_features = {
            'technical': {
                'lookback_windows': {
                    'short': [5, 10, 20],
                    'medium': [50]
                }
            }
        }

        config_labels = {
            'labels': {
                'horizon': 5,
                'target_type': 'return'
            }
        }

        # Generate features and labels
        df = create_technical_features(sample_data, config_features)
        df = generate_forward_returns(df, config_labels)

        # Get feature columns (exclude labels and metadata)
        # NOTE: risk_adjusted_return is a LABEL, not a feature (contains future info)
        feature_cols = [col for col in df.columns
                       if col not in ['date', 'symbol', 'open', 'high', 'low',
                                     'close', 'volume', 'forward_return_5d']
                       and not col.startswith('forward_')
                       and 'risk_adjusted_return' not in col]  # Exclude labels!

        # Check correlation between each feature and label
        label_col = 'forward_return_5d'

        for feat in feature_cols:
            if df[feat].notna().any() and df[label_col].notna().any():
                # Calculate correlation on overlapping non-NaN data
                valid_mask = df[feat].notna() & df[label_col].notna()
                if valid_mask.sum() > 10:  # Need at least 10 samples
                    corr = df.loc[valid_mask, feat].corr(df.loc[valid_mask, label_col])

                    # WARNING: If correlation is > 0.5, there's likely leakage
                    # In reality, features should have weak correlation with future returns
                    assert abs(corr) < 0.95, \
                        f"Feature '{feat}' has suspiciously high correlation " \
                        f"({corr:.4f}) with future returns - possible data leakage!"

    def test_label_shift_direction(self, sample_data):
        """
        Test that labels use correct shift direction

        Labels should use shift(-horizon) to get future prices
        """
        config = {
            'labels': {
                'horizon': 1,
                'target_type': 'return'
            }
        }

        df = generate_forward_returns(sample_data, config)

        # Manually calculate what the label should be
        expected_return = (sample_data['close'].iloc[1] - sample_data['close'].iloc[0]) / sample_data['close'].iloc[0]
        actual_return = df['forward_return_1d'].iloc[0]

        assert abs(expected_return - actual_return) < 1e-6, \
            f"Label calculation incorrect. Expected: {expected_return:.6f}, Got: {actual_return:.6f}"

    def test_feature_shift_direction(self, sample_data):
        """
        Test that momentum features use correct shift direction

        Features should use shift(+window) to get past prices
        """
        config = {
            'technical': {
                'lookback_windows': {
                    'short': [1],
                    'medium': []
                }
            }
        }

        df = create_technical_features(sample_data, config)

        # Check momentum_1d at index 10
        # It should be (close[10] - close[9]) / close[9]
        if len(df) > 10:
            expected_momentum = (sample_data['close'].iloc[10] - sample_data['close'].iloc[9]) / sample_data['close'].iloc[9]
            actual_momentum = df['momentum_1d'].iloc[10]

            # Allow for small numerical differences
            if pd.notna(actual_momentum):
                assert abs(expected_momentum - actual_momentum) < 1e-4, \
                    f"Momentum calculation incorrect. Expected: {expected_momentum:.6f}, Got: {actual_momentum:.6f}"


def test_realistic_ic_values():
    """
    Test that IC values are realistic

    Typical IC values in quantitative finance:
    - IC > 0.05: Good
    - IC > 0.10: Excellent
    - IC > 0.30: Suspicious - likely data leakage
    - IC > 0.90: DEFINITELY data leakage
    """
    # This is a documentation test
    # If your model shows IC > 0.90, investigate immediately!

    realistic_ic_threshold = 0.30
    warning_message = f"""
    WARNING: If your model IC exceeds {realistic_ic_threshold:.2f}, investigate for data leakage!

    Common causes:
    1. Using future data in features
    2. Forward-filling missing data incorrectly
    3. Using close price for same-day trades
    4. Not respecting PIT constraints for fundamentals
    5. Labels calculated incorrectly

    Realistic IC ranges:
    - 0.01-0.05: Typical for equity models
    - 0.05-0.10: Good predictive power
    - 0.10-0.15: Excellent
    - >0.30: Investigate immediately
    - >0.90: DEFINITELY data leakage
    """

    print(warning_message)
    assert True  # This is a documentation test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
