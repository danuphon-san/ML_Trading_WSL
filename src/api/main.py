"""
FastAPI backend for web dashboard
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import yaml

from src.database.models import init_database, SystemSignal, ActualTrade, Position
from sqlalchemy.orm import Session

# Initialize FastAPI
app = FastAPI(title="ML Trading Dashboard API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database
engine, SessionLocal = init_database()

# Config
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)


# Request/Response models
class TradeRequest(BaseModel):
    symbol: str
    side: str
    shares: float
    price: float
    is_manual: bool = False
    reason: Optional[str] = None


class BacktestRequest(BaseModel):
    start_date: str
    end_date: str
    initial_capital: float = 100000


class PositionResponse(BaseModel):
    symbol: str
    shares: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float


@app.get("/")
async def root():
    """Health check"""
    return {"status": "ok", "service": "ML Trading Dashboard"}


@app.get("/api/positions", response_model=List[PositionResponse])
async def get_positions():
    """Get current portfolio positions"""
    db = SessionLocal()
    try:
        positions = db.query(Position).all()
        return [
            PositionResponse(
                symbol=p.symbol,
                shares=p.shares,
                avg_cost=p.avg_cost or 0,
                current_price=p.current_price or 0,
                market_value=p.market_value or 0,
                unrealized_pnl=p.unrealized_pnl or 0
            )
            for p in positions
        ]
    finally:
        db.close()


@app.get("/api/signals")
async def get_signals(days: int = 7):
    """Get recent system signals"""
    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        signals = db.query(SystemSignal).filter(SystemSignal.date >= cutoff).all()

        return [
            {
                "id": s.id,
                "date": s.date.isoformat(),
                "symbol": s.symbol,
                "signal_type": s.signal_type,
                "ml_score": s.ml_score,
                "target_weight": s.target_weight,
                "target_price": s.target_price
            }
            for s in signals
        ]
    finally:
        db.close()


@app.get("/api/trades")
async def get_trades(days: int = 30):
    """Get trade history"""
    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        trades = db.query(ActualTrade).filter(ActualTrade.date >= cutoff).all()

        return [
            {
                "id": t.id,
                "date": t.date.isoformat(),
                "symbol": t.symbol,
                "side": t.side,
                "shares": t.shares,
                "execution_price": t.execution_price,
                "system_price": t.system_price,
                "price_deviation_bps": t.price_deviation_bps,
                "is_manual": t.is_manual_override,
                "reason": t.override_reason
            }
            for t in trades
        ]
    finally:
        db.close()


@app.post("/api/trade")
async def execute_trade(trade: TradeRequest):
    """Record manual trade execution"""
    db = SessionLocal()
    try:
        # Calculate price deviation if system price available
        deviation_bps = 0
        # TODO: Look up system signal for this symbol

        actual_trade = ActualTrade(
            date=datetime.utcnow(),
            symbol=trade.symbol,
            side=trade.side,
            shares=trade.shares,
            execution_price=trade.price,
            is_manual_override=trade.is_manual,
            override_reason=trade.reason
        )

        db.add(actual_trade)
        db.commit()

        return {"status": "success", "trade_id": actual_trade.id}
    finally:
        db.close()


@app.get("/api/reconciliation")
async def get_reconciliation(days: int = 7):
    """Get trade reconciliation data"""
    db = SessionLocal()
    try:
        from src.database.models import TradeReconciliation

        cutoff = datetime.utcnow() - timedelta(days=days)
        records = db.query(TradeReconciliation).filter(
            TradeReconciliation.date >= cutoff
        ).all()

        return [
            {
                "date": r.date.isoformat(),
                "symbol": r.symbol,
                "system_action": r.system_action,
                "actual_action": r.actual_action,
                "deviation_shares": r.deviation_shares,
                "deviation_bps": r.deviation_bps,
                "reason": r.reason
            }
            for r in records
        ]
    finally:
        db.close()


@app.post("/api/backtest")
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Run backtest (background task)"""
    # This would trigger a backtest run
    return {
        "status": "started",
        "message": f"Backtest queued for {request.start_date} to {request.end_date}"
    }


@app.get("/api/performance")
async def get_performance():
    """Get portfolio performance metrics"""
    db = SessionLocal()
    try:
        from src.database.models import PortfolioSnapshot

        snapshots = db.query(PortfolioSnapshot).order_by(
            PortfolioSnapshot.date.desc()
        ).limit(252).all()  # Last year

        if not snapshots:
            return {"message": "No performance data available"}

        equity_curve = [s.total_equity for s in reversed(snapshots)]
        dates = [s.date.isoformat() for s in reversed(snapshots)]

        # Calculate simple metrics
        returns = pd.Series(equity_curve).pct_change().dropna()
        total_return = (equity_curve[-1] / equity_curve[0] - 1) if equity_curve[0] > 0 else 0

        return {
            "dates": dates,
            "equity_curve": equity_curve,
            "total_return": total_return,
            "daily_return": returns.mean(),
            "volatility": returns.std() * (252 ** 0.5),
            "sharpe_ratio": (returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0
        }
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
