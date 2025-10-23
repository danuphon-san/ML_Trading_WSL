"""
Streamlit dashboard for ML Trading System
"""
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="ML Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Sidebar
st.sidebar.title("ML Trading Dashboard")
page = st.sidebar.selectbox(
    "Navigation",
    ["Portfolio Overview", "Signals & Trades", "Reconciliation", "Backtest", "Model Training"]
)

# Helper functions
def fetch_api(endpoint):
    """Fetch data from API"""
    try:
        response = requests.get(f"{API_URL}{endpoint}")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


# Page: Portfolio Overview
if page == "Portfolio Overview":
    st.title("📊 Portfolio Overview")

    col1, col2, col3, col4 = st.columns(4)

    # Fetch performance data
    perf = fetch_api("/api/performance")

    if perf and "equity_curve" in perf:
        with col1:
            st.metric("Total Return", f"{perf['total_return']:.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{perf.get('sharpe_ratio', 0):.2f}")
        with col3:
            st.metric("Volatility", f"{perf.get('volatility', 0):.2%}")
        with col4:
            current_equity = perf['equity_curve'][-1] if perf['equity_curve'] else 0
            st.metric("Current Equity", f"${current_equity:,.0f}")

        # Equity curve
        st.subheader("Equity Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=perf['dates'],
            y=perf['equity_curve'],
            mode='lines',
            name='Portfolio Value'
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No performance data available yet")

    # Current positions
    st.subheader("Current Positions")
    positions = fetch_api("/api/positions")

    if positions:
        df_positions = pd.DataFrame(positions)
        df_positions['unrealized_pnl_pct'] = (
            df_positions['unrealized_pnl'] / (df_positions['avg_cost'] * df_positions['shares']) * 100
        )

        st.dataframe(
            df_positions[[
                'symbol', 'shares', 'avg_cost', 'current_price',
                'market_value', 'unrealized_pnl', 'unrealized_pnl_pct'
            ]].style.format({
                'shares': '{:.2f}',
                'avg_cost': '${:.2f}',
                'current_price': '${:.2f}',
                'market_value': '${:,.2f}',
                'unrealized_pnl': '${:,.2f}',
                'unrealized_pnl_pct': '{:.2f}%'
            }),
            use_container_width=True
        )
    else:
        st.info("No positions currently held")


# Page: Signals & Trades
elif page == "Signals & Trades":
    st.title("📡 Signals & Trade Execution")

    tab1, tab2 = st.tabs(["System Signals", "Trade History"])

    with tab1:
        st.subheader("Recent System Signals")

        days = st.slider("Days to show", 1, 30, 7)
        signals = fetch_api(f"/api/signals?days={days}")

        if signals:
            df_signals = pd.DataFrame(signals)
            st.dataframe(
                df_signals[[
                    'date', 'symbol', 'signal_type', 'ml_score',
                    'target_weight', 'target_price'
                ]].sort_values('date', ascending=False),
                use_container_width=True
            )
        else:
            st.info("No signals available")

    with tab2:
        st.subheader("Trade History")

        days = st.slider("Days to show", 1, 90, 30, key="trades_days")
        trades = fetch_api(f"/api/trades?days={days}")

        if trades:
            df_trades = pd.DataFrame(trades)

            # Highlight manual trades
            st.dataframe(
                df_trades[[
                    'date', 'symbol', 'side', 'shares',
                    'execution_price', 'price_deviation_bps', 'is_manual'
                ]].sort_values('date', ascending=False),
                use_container_width=True
            )

            # Manual trade entry
            st.subheader("Record Manual Trade")
            with st.form("manual_trade"):
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    symbol = st.text_input("Symbol")
                with col2:
                    side = st.selectbox("Side", ["buy", "sell"])
                with col3:
                    shares = st.number_input("Shares", min_value=0.0, step=1.0)
                with col4:
                    price = st.number_input("Price", min_value=0.0, step=0.01)

                reason = st.text_area("Reason for manual override")

                submitted = st.form_submit_button("Record Trade")

                if submitted and symbol and shares > 0:
                    trade_data = {
                        "symbol": symbol,
                        "side": side,
                        "shares": shares,
                        "price": price,
                        "is_manual": True,
                        "reason": reason
                    }

                    try:
                        response = requests.post(f"{API_URL}/api/trade", json=trade_data)
                        if response.status_code == 200:
                            st.success("Trade recorded successfully")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Failed to record trade: {e}")
        else:
            st.info("No trades recorded")


# Page: Reconciliation
elif page == "Reconciliation":
    st.title("🔍 Trade Reconciliation")

    st.markdown("""
    Track deviations between system signals and actual executions.
    This helps identify manual overrides, price slippage, and execution quality.
    """)

    days = st.slider("Days to show", 1, 30, 7)
    recon = fetch_api(f"/api/reconciliation?days={days}")

    if recon:
        df_recon = pd.DataFrame(recon)

        st.dataframe(
            df_recon[[
                'date', 'symbol', 'system_action', 'actual_action',
                'deviation_shares', 'deviation_bps', 'reason'
            ]].sort_values('date', ascending=False),
            use_container_width=True
        )

        # Summary stats
        st.subheader("Reconciliation Summary")
        col1, col2, col3 = st.columns(3)

        with col1:
            n_deviations = len(df_recon[df_recon['deviation_bps'].abs() > 5])
            st.metric("Trades with >5bps deviation", n_deviations)

        with col2:
            avg_deviation = df_recon['deviation_bps'].abs().mean()
            st.metric("Avg Price Deviation", f"{avg_deviation:.1f} bps")

        with col3:
            manual_overrides = len(df_recon[df_recon['reason'].str.contains('manual', case=False, na=False)])
            st.metric("Manual Overrides", manual_overrides)

    else:
        st.info("No reconciliation data available")


# Page: Backtest
elif page == "Backtest":
    st.title("⚡ Run Backtest")

    with st.form("backtest_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        with col3:
            initial_capital = st.number_input("Initial Capital", value=100000, step=10000)

        submitted = st.form_submit_button("Run Backtest")

        if submitted:
            backtest_data = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "initial_capital": initial_capital
            }

            try:
                response = requests.post(f"{API_URL}/api/backtest", json=backtest_data)
                if response.status_code == 200:
                    st.success("Backtest started! Results will appear when complete.")
            except Exception as e:
                st.error(f"Failed to start backtest: {e}")


# Page: Model Training
elif page == "Model Training":
    st.title("🤖 Model Training & Evaluation")

    st.markdown("""
    Monitor ML model performance and trigger retraining.
    """)

    # Model metrics (placeholder)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Current Model IC", "0.08")
    with col2:
        st.metric("Rank IC", "0.12")
    with col3:
        st.metric("Last Trained", "2024-10-15")

    # Feature importance (placeholder)
    st.subheader("Feature Importance")
    st.info("Feature importance visualization will appear here")

    if st.button("Trigger Model Retraining"):
        st.info("Model retraining started (background task)")


# Footer
st.sidebar.markdown("---")
st.sidebar.info("ML Trading System v1.0")
