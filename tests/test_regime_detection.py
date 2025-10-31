"""
Test script for regime detection functionality

Tests:
1. Standalone regime detection on SPY
2. Regime-aware portfolio construction
3. Regime feature engineering
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from loguru import logger

from src.portfolio.regime_detection import RegimeDetector, run_regime_detection
from src.portfolio.construct import construct_portfolio, _detect_current_regime
from src.features.ta_features import TechnicalFeatures
from utils.config_loader import load_config


def test_regime_detection_standalone():
    """Test 1: Standalone regime detection on SPY benchmark"""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Standalone Regime Detection on SPY")
    logger.info("="*80)

    # Load config
    config = load_config()

    # Load SPY data from parquet if available
    parquet_dir = config['data']['parquet']
    spy_file = f"{parquet_dir}/1d/SPY.parquet"

    if not os.path.exists(spy_file):
        logger.warning(f"SPY data not found at {spy_file}")
        logger.info("Attempting to download SPY data for testing...")
        try:
            import yfinance as yf
            spy = yf.download('SPY', start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'), progress=False)
            spy_data = pd.DataFrame({
                'date': spy.index,
                'symbol': 'SPY',
                'close': spy['Close'].values
            })
        except Exception as e:
            logger.error(f"Failed to download SPY data: {e}")
            return False
    else:
        spy_data = pd.read_parquet(spy_file)
        spy_data['symbol'] = 'SPY'

    # Sort by date
    spy_data = spy_data.sort_values('date').reset_index(drop=True)

    logger.info(f"Loaded {len(spy_data)} days of SPY data from {spy_data['date'].min()} to {spy_data['date'].max()}")

    # Initialize regime detector
    detector = RegimeDetector(config)

    # Detect regimes
    regime_df = detector.detect(
        prices=spy_data['close'],
        dates=spy_data['date']
    )

    logger.info(f"\n📊 Regime Detection Results:")
    logger.info(f"Total periods analyzed: {len(regime_df)}")

    # Count regime distribution
    regime_counts = regime_df['regime'].value_counts().sort_index()
    regime_names = {0: 'Risk-Off', 1: 'Normal', 2: 'Risk-On'}

    for regime_id, count in regime_counts.items():
        pct = count / len(regime_df) * 100
        logger.info(f"  {regime_names[regime_id]:10s}: {count:5d} days ({pct:5.1f}%)")

    # Get current regime
    current_regime = detector.get_current_regime(regime_df)
    logger.info(f"\n🎯 Current Regime: {current_regime['regime_name']}")
    logger.info(f"   Risk Multiplier: {current_regime['risk_multiplier']:.2f}x")

    if 'volatility' in current_regime:
        logger.info(f"   Volatility: {current_regime['volatility']:.2%}")
        logger.info(f"   Drawdown: {current_regime['drawdown']:.2%}")

    # Show recent regime transitions
    logger.info(f"\n📅 Recent Regime History (last 20 days):")
    recent = regime_df.tail(20)[['date', 'regime', 'volatility', 'drawdown', 'risk_multiplier']]
    recent['regime_name'] = recent['regime'].map(regime_names)
    logger.info(f"\n{recent.to_string(index=False)}")

    # Save results
    output_file = "data/reports/regime_detection_test.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    regime_df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Regime detection results saved to {output_file}")

    return True


def test_regime_features():
    """Test 2: Regime features in technical indicators"""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Regime Features in Technical Indicators")
    logger.info("="*80)

    # Load config
    config = load_config()

    # Load sample stock data (use SPY for testing)
    parquet_dir = config['data']['parquet']
    spy_file = f"{parquet_dir}/1d/SPY.parquet"

    if not os.path.exists(spy_file):
        logger.warning(f"SPY data not found at {spy_file}")
        try:
            import yfinance as yf
            spy = yf.download('SPY', start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'), progress=False)
            spy_data = pd.DataFrame({
                'date': spy.index,
                'symbol': 'SPY',
                'open': spy['Open'].values,
                'high': spy['High'].values,
                'low': spy['Low'].values,
                'close': spy['Close'].values,
                'volume': spy['Volume'].values
            })
        except Exception as e:
            logger.error(f"Failed to download SPY data: {e}")
            return False
    else:
        spy_data = pd.read_parquet(spy_file)
        spy_data['symbol'] = 'SPY'

    # Initialize feature engineer
    ta_features = TechnicalFeatures(config)

    # Compute features (this should include regime features now)
    logger.info(f"Computing technical features for SPY ({len(spy_data)} days)...")
    features_df = ta_features.compute_features({'SPY': spy_data})

    # Check if regime features exist
    regime_feature_cols = [
        'vol_regime', 'trend_strength', 'drawdown_pct', 'dist_from_200d_high',
        'above_200_sma', 'momentum_consensus', 'vol_expansion', 'crisis_indicator',
        'recovery_indicator'
    ]

    logger.info(f"\n📊 Checking for regime features:")
    found_features = []
    missing_features = []

    for col in regime_feature_cols:
        if col in features_df.columns:
            found_features.append(col)
            # Show statistics for this feature
            valid_values = features_df[col].dropna()
            logger.info(f"  ✓ {col:25s}: {len(valid_values):5d} valid values, "
                       f"mean={valid_values.mean():7.3f}, "
                       f"std={valid_values.std():7.3f}")
        else:
            missing_features.append(col)
            logger.warning(f"  ✗ {col:25s}: NOT FOUND")

    # Show recent regime feature values
    if found_features:
        logger.info(f"\n📅 Recent Regime Feature Values (last 10 days):")
        recent_cols = ['date'] + [f for f in regime_feature_cols if f in features_df.columns]
        recent = features_df[recent_cols].tail(10)
        logger.info(f"\n{recent.to_string(index=False)}")

        # Save results
        output_file = "data/reports/regime_features_test.csv"
        features_df.to_csv(output_file, index=False)
        logger.info(f"\n✓ Regime features saved to {output_file}")

        return len(missing_features) == 0
    else:
        logger.error("❌ No regime features found!")
        return False


def test_regime_aware_portfolio():
    """Test 3: Regime-aware portfolio construction"""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Regime-Aware Portfolio Construction")
    logger.info("="*80)

    # Load config
    config = load_config()

    # Create mock scored data (5 stocks with ML scores)
    mock_scores = pd.DataFrame({
        'date': ['2025-10-29'] * 5,
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
        'ml_score': [0.85, 0.82, 0.78, 0.75, 0.72]
    })
    mock_scores['date'] = pd.to_datetime(mock_scores['date'])

    # Create mock price panel (need historical data for covariance)
    logger.info("Creating mock price panel...")
    dates = pd.date_range(end='2025-10-29', periods=252, freq='D')

    mock_prices_list = []
    for symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'SPY']:
        # Generate synthetic price data
        np.random.seed(hash(symbol) % 2**32)
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = 100 * (1 + returns).cumprod()

        mock_prices_list.append(pd.DataFrame({
            'date': dates,
            'symbol': symbol,
            'close': prices
        }))

    price_panel = pd.concat(mock_prices_list, ignore_index=True)

    logger.info(f"Mock data created: {len(mock_scores)} stocks, {len(dates)} days of price history")

    # Test 3a: Portfolio WITHOUT regime adaptation
    logger.info("\n--- Test 3a: Portfolio WITHOUT Regime Adaptation ---")
    weights_no_regime = construct_portfolio(
        scored_df=mock_scores,
        price_panel=price_panel,
        config=config,
        enable_regime_adaptation=False
    )

    logger.info(f"Portfolio weights (no regime adaptation):")
    for symbol, weight in sorted(weights_no_regime.items(), key=lambda x: -x[1]):
        logger.info(f"  {symbol}: {weight:.2%}")
    logger.info(f"Total weight: {sum(weights_no_regime.values()):.2%}")

    # Test 3b: Portfolio WITH regime adaptation
    logger.info("\n--- Test 3b: Portfolio WITH Regime Adaptation ---")
    weights_with_regime = construct_portfolio(
        scored_df=mock_scores,
        price_panel=price_panel,
        config=config,
        enable_regime_adaptation=True
    )

    logger.info(f"Portfolio weights (WITH regime adaptation):")
    for symbol, weight in sorted(weights_with_regime.items(), key=lambda x: -x[1]):
        logger.info(f"  {symbol}: {weight:.2%}")
    logger.info(f"Total weight: {sum(weights_with_regime.values()):.2%}")

    # Compare
    logger.info("\n📊 Comparison:")
    logger.info(f"  Without regime: {len(weights_no_regime)} positions, total={sum(weights_no_regime.values()):.2%}")
    logger.info(f"  With regime:    {len(weights_with_regime)} positions, total={sum(weights_with_regime.values()):.2%}")

    cash_allocation = 1.0 - sum(weights_with_regime.values())
    if abs(cash_allocation) > 0.01:
        logger.info(f"  💰 Implicit cash: {cash_allocation:.2%}")

    return True


def main():
    """Run all tests"""
    logger.info("="*80)
    logger.info("REGIME DETECTION TEST SUITE")
    logger.info("="*80)

    results = {}

    # Test 1: Standalone regime detection
    try:
        results['test_1'] = test_regime_detection_standalone()
    except Exception as e:
        logger.error(f"Test 1 failed with error: {e}")
        results['test_1'] = False

    # Test 2: Regime features
    try:
        results['test_2'] = test_regime_features()
    except Exception as e:
        logger.error(f"Test 2 failed with error: {e}")
        results['test_2'] = False

    # Test 3: Regime-aware portfolio
    try:
        results['test_3'] = test_regime_aware_portfolio()
    except Exception as e:
        logger.error(f"Test 3 failed with error: {e}")
        results['test_3'] = False

    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        logger.info("\n🎉 All tests passed!")
    else:
        logger.warning("\n⚠️  Some tests failed. Please review the logs above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
