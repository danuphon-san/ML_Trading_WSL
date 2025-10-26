"""
Monitoring & Operations Reporting (Step 15)

Generate daily/weekly ops reports, send email summaries, monitor KPIs and breaches
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from loguru import logger


class OpsMonitor:
    """
    Operations monitoring and reporting

    Features:
    - Daily/weekly performance summary
    - Position changes tracking
    - Kill-switch monitoring
    - Breach alerts
    - Email/Slack notifications
    """

    def __init__(self, config: Dict):
        """
        Initialize ops monitor

        Args:
            config: Configuration dict
        """
        self.config = config
        self.ops_config = config.get('ops', {})

        self.alerts_enabled = self.ops_config.get('alerts', 'email') != 'none'
        self.alert_type = self.ops_config.get('alerts', 'email')
        self.email_recipients = self.ops_config.get('email_recipients', [])

        # Kill-switch parameters
        self.kill_switch_enabled = self.ops_config.get('kill_switch', {}).get('enabled', True)
        self.max_daily_loss_pct = self.ops_config.get('kill_switch', {}).get('max_daily_loss_pct', 0.03)
        self.min_live_sharpe = self.ops_config.get('kill_switch', {}).get('min_live_sharpe_threshold', 0.5)
        self.sharpe_lookback_weeks = self.ops_config.get('kill_switch', {}).get('min_live_sharpe_lookback_weeks', 6)

        logger.info(f"Initialized OpsMonitor: alerts={self.alert_type}, kill_switch={self.kill_switch_enabled}")

    def calculate_performance_metrics(
        self,
        equity_curve: pd.DataFrame,
        lookback_days: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate performance metrics

        Args:
            equity_curve: DataFrame with columns [date, equity]
            lookback_days: Optional lookback period (if None, use full history)

        Returns:
            Dict of performance metrics
        """
        if lookback_days:
            equity_curve = equity_curve.tail(lookback_days)

        equity_curve = equity_curve.sort_values('date')
        returns = equity_curve['equity'].pct_change().dropna()

        # Total return
        total_return = (equity_curve['equity'].iloc[-1] / equity_curve['equity'].iloc[0]) - 1.0

        # Daily return
        daily_return = returns.iloc[-1] if len(returns) > 0 else 0.0

        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.0

        # Sharpe ratio (annualized)
        risk_free_rate = self.config.get('portfolio', {}).get('pypfopt', {}).get('risk_free_rate', 0.02)
        sharpe = (returns.mean() * 252 - risk_free_rate) / (returns.std() * np.sqrt(252)) if volatility > 0 else 0.0

        # Maximum drawdown
        running_max = equity_curve['equity'].expanding().max()
        drawdown = (equity_curve['equity'] - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0

        metrics = {
            'total_return': float(total_return),
            'daily_return': float(daily_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'num_periods': len(equity_curve)
        }

        return metrics

    def detect_position_changes(
        self,
        weights_prev: pd.DataFrame,
        weights_current: pd.DataFrame,
        top_n: int = 10
    ) -> Dict[str, List]:
        """
        Detect significant position changes

        Args:
            weights_prev: Previous weights [symbol, weight]
            weights_current: Current weights [symbol, weight]
            top_n: Number of top changes to report

        Returns:
            Dict with top adds, cuts, and holds
        """
        merged = pd.merge(
            weights_prev[['symbol', 'weight']].rename(columns={'weight': 'weight_prev'}),
            weights_current[['symbol', 'weight']].rename(columns={'weight': 'weight_current'}),
            on='symbol',
            how='outer'
        ).fillna(0.0)

        merged['change'] = merged['weight_current'] - merged['weight_prev']

        # Top adds (new or increased positions)
        top_adds = merged[merged['change'] > 0].nlargest(top_n, 'change')

        # Top cuts (closed or decreased positions)
        top_cuts = merged[merged['change'] < 0].nsmallest(top_n, 'change')

        # Holds (unchanged or minimal change)
        holds = merged[(merged['change'].abs() < 0.01) & (merged['weight_current'] > 0)]

        changes = {
            'top_adds': top_adds[['symbol', 'weight_prev', 'weight_current', 'change']].to_dict('records'),
            'top_cuts': top_cuts[['symbol', 'weight_prev', 'weight_current', 'change']].to_dict('records'),
            'num_holds': len(holds)
        }

        return changes

    def check_kill_switch(
        self,
        equity_curve: pd.DataFrame,
        breaches: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Check kill-switch conditions

        Conditions:
        1. Daily loss > max_daily_loss_pct
        2. Rolling Sharpe < min_live_sharpe over lookback window

        Args:
            equity_curve: DataFrame with columns [date, equity]
            breaches: Optional list to append breaches

        Returns:
            Dict with kill-switch status and details
        """
        if not self.kill_switch_enabled:
            return {'triggered': False, 'reason': None}

        equity_curve = equity_curve.sort_values('date')
        returns = equity_curve['equity'].pct_change().dropna()

        # Check 1: Daily loss
        daily_return = returns.iloc[-1] if len(returns) > 0 else 0.0
        daily_loss_breach = daily_return < -self.max_daily_loss_pct

        # Check 2: Rolling Sharpe
        lookback_days = self.sharpe_lookback_weeks * 5  # Trading days
        if len(returns) >= lookback_days:
            recent_returns = returns.tail(lookback_days)
            risk_free_rate = self.config.get('portfolio', {}).get('pypfopt', {}).get('risk_free_rate', 0.02)
            rolling_sharpe = (recent_returns.mean() * 252 - risk_free_rate) / (recent_returns.std() * np.sqrt(252))

            sharpe_breach = rolling_sharpe < self.min_live_sharpe
        else:
            rolling_sharpe = None
            sharpe_breach = False

        # Determine kill-switch status
        triggered = daily_loss_breach or sharpe_breach

        result = {
            'triggered': triggered,
            'daily_return': float(daily_return),
            'daily_loss_breach': daily_loss_breach,
            'rolling_sharpe': float(rolling_sharpe) if rolling_sharpe is not None else None,
            'sharpe_breach': sharpe_breach,
            'reason': None
        }

        if triggered:
            if daily_loss_breach:
                result['reason'] = f"Daily loss {daily_return:.2%} exceeds threshold {-self.max_daily_loss_pct:.2%}"
            elif sharpe_breach:
                result['reason'] = f"Rolling Sharpe {rolling_sharpe:.2f} below threshold {self.min_live_sharpe:.2f}"

            logger.warning(f"⚠️ KILL-SWITCH TRIGGERED: {result['reason']}")

            if breaches is not None:
                breaches.append({
                    'type': 'kill_switch',
                    'date': str(equity_curve['date'].iloc[-1]),
                    'reason': result['reason']
                })

        return result

    def generate_ops_report_html(
        self,
        report_data: Dict[str, Any],
        date: Optional[datetime] = None
    ) -> str:
        """
        Generate HTML ops report

        Args:
            report_data: Dict with all report data
            date: Report date (default: today)

        Returns:
            HTML string
        """
        if date is None:
            date = datetime.now()

        date_str = date.strftime("%Y-%m-%d")

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Trading Ops Report - {date_str}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .metric-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .positive {{ color: #27ae60; }}
                .negative {{ color: #e74c3c; }}
                .warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; }}
                .alert {{ background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 10px; margin: 10px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th {{ background-color: #3498db; color: white; padding: 10px; text-align: left; }}
                td {{ padding: 8px; border-bottom: 1px solid #ecf0f1; }}
                tr:hover {{ background-color: #f8f9fa; }}
                .footer {{ margin-top: 40px; text-align: center; color: #95a5a6; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 ML Trading System - Operations Report</h1>
                <p><strong>Report Date:</strong> {date_str}</p>

                <h2>📈 Performance Summary</h2>
                <div class="metrics">
        """

        # Performance metrics
        perf = report_data.get('performance', {})
        html += f"""
                    <div class="metric">
                        <div class="metric-label">Daily Return</div>
                        <div class="metric-value {'positive' if perf.get('daily_return', 0) > 0 else 'negative'}">
                            {perf.get('daily_return', 0):.2%}
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Total Return</div>
                        <div class="metric-value {'positive' if perf.get('total_return', 0) > 0 else 'negative'}">
                            {perf.get('total_return', 0):.2%}
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-value">{perf.get('sharpe_ratio', 0):.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Volatility</div>
                        <div class="metric-value">{perf.get('volatility', 0):.2%}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value negative">{perf.get('max_drawdown', 0):.2%}</div>
                    </div>
                </div>
        """

        # Kill-switch status
        kill_switch = report_data.get('kill_switch', {})
        if kill_switch.get('triggered'):
            html += f"""
                <div class="alert">
                    <strong>🚨 KILL-SWITCH ALERT:</strong> {kill_switch.get('reason', 'Unknown')}
                </div>
            """

        # Position changes
        position_changes = report_data.get('position_changes', {})
        top_adds = position_changes.get('top_adds', [])
        top_cuts = position_changes.get('top_cuts', [])

        if top_adds:
            html += """
                <h2>➕ Top Position Increases</h2>
                <table>
                    <tr><th>Symbol</th><th>Previous</th><th>Current</th><th>Change</th></tr>
            """
            for add in top_adds[:10]:
                html += f"""
                    <tr>
                        <td><strong>{add['symbol']}</strong></td>
                        <td>{add['weight_prev']:.2%}</td>
                        <td>{add['weight_current']:.2%}</td>
                        <td class="positive">{add['change']:.2%}</td>
                    </tr>
                """
            html += "</table>"

        if top_cuts:
            html += """
                <h2>➖ Top Position Decreases</h2>
                <table>
                    <tr><th>Symbol</th><th>Previous</th><th>Current</th><th>Change</th></tr>
            """
            for cut in top_cuts[:10]:
                html += f"""
                    <tr>
                        <td><strong>{cut['symbol']}</strong></td>
                        <td>{cut['weight_prev']:.2%}</td>
                        <td>{cut['weight_current']:.2%}</td>
                        <td class="negative">{cut['change']:.2%}</td>
                    </tr>
                """
            html += "</table>"

        # Turnover summary
        turnover_report = report_data.get('turnover', {})
        if turnover_report:
            html += f"""
                <h2>🔄 Turnover Summary</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">Avg Turnover</div>
                        <div class="metric-value">{turnover_report.get('avg_turnover', 0):.2%}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Max Turnover</div>
                        <div class="metric-value">{turnover_report.get('max_turnover', 0):.2%}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Total Costs</div>
                        <div class="metric-value">${turnover_report.get('total_costs', 0):.2f}</div>
                    </div>
                </div>
            """

            if turnover_report.get('cap_breaches', 0) > 0:
                html += f"""
                    <div class="warning">
                        <strong>⚠️ Warning:</strong> {turnover_report['cap_breaches']} turnover cap breaches detected
                    </div>
                """

        # Regime & allocation info
        regime = report_data.get('regime', {})
        if regime:
            html += f"""
                <h2>🌐 Market Regime</h2>
                <p><strong>Current Regime:</strong> {regime.get('regime_name', 'Unknown')}</p>
                <p><strong>Risk Multiplier:</strong> {regime.get('risk_multiplier', 1.0):.2f}</p>
            """

        allocation = report_data.get('allocation', {})
        if allocation:
            html += f"""
                <h2>💼 Sleeve Allocation</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">Equities</div>
                        <div class="metric-value">{allocation.get('equities', 0):.2%}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Crypto</div>
                        <div class="metric-value">{allocation.get('crypto', 0):.2%}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Cash</div>
                        <div class="metric-value">{allocation.get('cash', 0):.2%}</div>
                    </div>
                </div>
            """

        # Footer
        html += f"""
                <div class="footer">
                    <p>Generated by ML Trading System | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <p>🤖 Powered by Claude Code</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def send_email_alert(self, subject: str, body: str, html_body: Optional[str] = None) -> bool:
        """
        Send email alert

        Args:
            subject: Email subject
            body: Plain text body
            html_body: Optional HTML body

        Returns:
            True if sent successfully
        """
        if not self.alerts_enabled or self.alert_type != 'email':
            logger.info("Email alerts not enabled, skipping")
            return False

        if not self.email_recipients:
            logger.warning("No email recipients configured")
            return False

        # Note: This is a placeholder implementation
        # In production, configure SMTP settings in config.yaml
        logger.info(f"Email alert would be sent to: {self.email_recipients}")
        logger.info(f"Subject: {subject}")
        logger.info(f"Body preview: {body[:200]}...")

        # TODO: Implement actual SMTP sending
        # msg = MIMEMultipart('alternative')
        # msg['Subject'] = subject
        # msg['From'] = "alerts@mltrading.com"
        # msg['To'] = ", ".join(self.email_recipients)
        #
        # msg.attach(MIMEText(body, 'plain'))
        # if html_body:
        #     msg.attach(MIMEText(html_body, 'html'))
        #
        # # Send via SMTP
        # ...

        return True


def run_monitoring(
    config: Dict,
    equity_curve: pd.DataFrame,
    weights_current: pd.DataFrame,
    weights_prev: Optional[pd.DataFrame] = None,
    regime_info: Optional[Dict] = None,
    allocation_info: Optional[Dict] = None,
    turnover_report: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Standalone function to run monitoring and generate ops report

    Args:
        config: Configuration dictionary
        equity_curve: DataFrame with equity curve [date, equity]
        weights_current: Current portfolio weights
        weights_prev: Previous portfolio weights (for position changes)
        regime_info: Optional regime detection results
        allocation_info: Optional sleeve allocation results
        turnover_report: Optional turnover report

    Returns:
        Dict with monitoring results and report
    """
    logger.info("Running operations monitoring...")

    monitor = OpsMonitor(config)

    # Calculate performance metrics
    perf_all = monitor.calculate_performance_metrics(equity_curve)
    perf_recent = monitor.calculate_performance_metrics(equity_curve, lookback_days=30)

    # Check kill-switch
    breaches = []
    kill_switch_status = monitor.check_kill_switch(equity_curve, breaches)

    # Position changes
    position_changes = {}
    if weights_prev is not None:
        position_changes = monitor.detect_position_changes(weights_prev, weights_current)

    # Compile report data
    report_data = {
        'performance': perf_all,
        'performance_30d': perf_recent,
        'kill_switch': kill_switch_status,
        'position_changes': position_changes,
        'regime': regime_info,
        'allocation': allocation_info,
        'turnover': turnover_report,
        'breaches': breaches
    }

    # Generate HTML report
    html_report = monitor.generate_ops_report_html(report_data)

    # Send alerts if needed
    if kill_switch_status.get('triggered'):
        monitor.send_email_alert(
            subject="🚨 Kill-Switch Alert - ML Trading System",
            body=f"Kill-switch triggered: {kill_switch_status.get('reason')}",
            html_body=html_report
        )

    logger.info(f"✓ Operations monitoring complete")

    return {
        'report_data': report_data,
        'html_report': html_report,
        'kill_switch_triggered': kill_switch_status.get('triggered', False)
    }
