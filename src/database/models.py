"""
Database models for trade tracking and reconciliation
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class SystemSignal(Base):
    """System-generated trading signals"""
    __tablename__ = 'system_signals'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    symbol = Column(String(20), nullable=False)
    signal_type = Column(String(20))  # buy, sell, hold
    ml_score = Column(Float)
    target_weight = Column(Float)
    target_price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class ActualTrade(Base):
    """Actual executed trades"""
    __tablename__ = 'actual_trades'

    id = Column(Integer, primary_key=True)
    signal_id = Column(Integer)  # Link to system signal
    date = Column(DateTime, nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10))  # buy, sell
    shares = Column(Float, nullable=False)
    execution_price = Column(Float, nullable=False)
    system_price = Column(Float)  # Price from system signal
    price_deviation_bps = Column(Float)  # Execution vs system
    commission = Column(Float, default=0.0)
    is_manual_override = Column(Boolean, default=False)
    override_reason = Column(String(500))
    executed_at = Column(DateTime, default=datetime.utcnow)


class Position(Base):
    """Current portfolio positions"""
    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, unique=True)
    shares = Column(Float, nullable=False)
    avg_cost = Column(Float)
    current_price = Column(Float)
    market_value = Column(Float)
    unrealized_pnl = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PortfolioSnapshot(Base):
    """Daily portfolio snapshots"""
    __tablename__ = 'portfolio_snapshots'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    total_equity = Column(Float, nullable=False)
    cash = Column(Float)
    positions_value = Column(Float)
    daily_return = Column(Float)
    daily_pnl = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class TradeReconciliation(Base):
    """Trade reconciliation records"""
    __tablename__ = 'trade_reconciliation'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    symbol = Column(String(20), nullable=False)
    system_action = Column(String(20))  # buy, sell, hold
    actual_action = Column(String(20))  # buy, sell, hold, none
    system_shares = Column(Float)
    actual_shares = Column(Float)
    deviation_shares = Column(Float)
    system_price = Column(Float)
    actual_price = Column(Float)
    deviation_bps = Column(Float)
    reason = Column(String(500))  # Reason for deviation
    created_at = Column(DateTime, default=datetime.utcnow)


def init_database(db_url: str = 'sqlite:///data/trading.db'):
    """Initialize database"""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine)
