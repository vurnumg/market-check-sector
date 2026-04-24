from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import math

import pandas as pd
import yfinance as yf


# -------------------------------
# Settings
# -------------------------------
RISK_PER_TRADE = 100.0  # Fixed £ risk per trade
PORTFOLIO_FILE = "portfolio.csv"
UPDATED_PORTFOLIO_FILE = "portfolio.csv"
PRICE_DECIMALS = 2

# Price scale used for sizing and portfolio valuation
# 1.0  = use price as-is
# 0.01 = divide by 100 because Yahoo price is effectively in pence for cash calcs
PRICE_SCALE = {
    "IITU": 0.01,
    "IESU": 0.01,
    "UIFS": 0.01,
    "IHCU": 0.01,
    "IISU": 0.01,
    "ICDU": 0.01,
    "ICSU": 0.01,
    "IUSU": 0.01,
    "IMSU": 0.01,
    "EWSP": 1,
}

# Yahoo symbols where the latest close can occasionally be returned in pence
PENCE_ANOMALY_SYMBOLS = set()


# -------------------------------
# Watchlist (Trading 212 tickers)
# -------------------------------
WATCHLIST: Dict[str, str] = {
    "IITU": "IITU.L",
    "IESU": "IESU.L",
    "UIFS": "UIFS.L",
    "IHCU": "IHCU.L",
    "IISU": "IISU.L",
    "ICDU": "ICDU.L",
    "ICSU": "ICSU.L",
    "IUSU": "IUSU.L",
    "IMSU": "IMSU.L",
    "EWSP": "EWSP.L",
}

# Tickers that should use CLOSE-based channels
CLOSE_ONLY = set()


@dataclass
class SignalResult:
    name: str
    symbol: str
    close: float
    prior_100_high: float
    prior_50_low: float
    pct_to_breakout: float
    pct_to_sale: float
    entry_trigger: bool
    exit_trigger: bool
    status: str
    pct_change_from_yesterday: float
    risk_per_share: float | None
    position_size: int | None
    capital_required: float | None


# -------------------------------
# Helpers
# -------------------------------

def normalise_price(value: float | int | None) -> float:
    return round(float(value), PRICE_DECIMALS)


# -------------------------------
# Data download & cleaning
# -------------------------------

def download_ohlc(symbol: str, period: str = "18mo") -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {symbol}: {missing}")

    df = df.dropna(subset=required).copy()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.tail(250)

    if len(df) < 120:
        raise ValueError(f"Not enough data for {symbol}")

    return df


# -------------------------------
# Symbol-specific fixes
# -------------------------------

def apply_symbol_fixes(name: str, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if symbol in PENCE_ANOMALY_SYMBOLS:
        last_close = df.iloc[-1]["Close"]
        recent_closes = df["Close"].iloc[-10:-1]

        if len(recent_closes) > 0:
            median_close = recent_closes.median()

            if median_close > 0 and last_close / median_close > 50:
                df.iloc[-1, df.columns.get_loc("Close")] = last_close / 100.0

    return df


# -------------------------------
# Signal calculation
# -------------------------------

def compute_channels(name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if name in CLOSE_ONLY:
        df["prior_100_high"] = df["Close"].rolling(100).max().shift(1)
        df["prior_50_low"] = df["Close"].rolling(50).min().shift(1)
    else:
        df["prior_100_high"] = df["High"].rolling(100).max().shift(1)
        df["prior_50_low"] = df["Low"].rolling(50).min().shift(1)

    df["prior_100_high"] = df["prior_100_high"].round(PRICE_DECIMALS)
    df["prior_50_low"] = df["prior_50_low"].round(PRICE_DECIMALS)

    return df


def convert_price_for_cash_calcs(name: str, value: float) -> float:
    """
    Converts a displayed market price into the cash value used for:
    - position sizing
    - entry value
    - stop value
    - current value
    - P&L
    """
    scale = PRICE_SCALE.get(name, 1.0)
    return value * scale


def calculate_position_size(
    name: str,
    close: float,
    prior_50_low: float,
    status: str
) -> tuple[float | None, int | None, float | None]:
    if status != "BUY":
        return None, None, None

    close_for_sizing = convert_price_for_cash_calcs(name, close)
    stop_for_sizing = convert_price_for_cash_calcs(name, prior_50_low)

    risk_per_share = normalise_price(close_for_sizing - stop_for_sizing)

    if risk_per_share <= 0:
        return None, None, None

    position_size = math.floor(RISK_PER_TRADE / risk_per_share)

    if position_size <= 0:
        return risk_per_share, 0, 0.0

    capital_required = normalise_price(position_size * close_for_sizing)
    return risk_per_share, position_size, capital_required


def get_signal(name: str, symbol: str) -> SignalResult:
    df = download_ohlc(symbol)
    df = apply_symbol_fixes(name, symbol, df)
    df = compute_channels(name, df)

    df["daily_pct_change"] = df["Close"].pct_change() * 100

    last = df.iloc[-1]

    close = normalise_price(last["Close"])
    prior_100_high = normalise_price(last["prior_100_high"])
    prior_50_low = normalise_price(last["prior_50_low"])
    pct_change_from_yesterday = (
        float(last["daily_pct_change"]) if pd.notna(last["daily_pct_change"]) else 0.0
    )

    if math.isnan(prior_100_high) or math.isnan(prior_50_low):
        raise ValueError(f"Rolling values unavailable for {symbol}")

    entry_trigger = close > prior_100_high
    exit_trigger = close < prior_50_low

    pct_to_breakout = ((prior_100_high - close) / close) * 100
    pct_to_sale = ((close - prior_50_low) / prior_50_low) * 100

    if entry_trigger:
        status = "BUY"
    elif exit_trigger:
        status = "SELL"
    else:
        status = "HOLD"

    risk_per_share, position_size, capital_required = calculate_position_size(
        name=name,
        close=close,
        prior_50_low=prior_50_low,
        status=status,
    )

    return SignalResult(
        name=name,
        symbol=symbol,
        close=close,
        prior_100_high=prior_100_high,
        prior_50_low=prior_50_low,
        pct_to_breakout=pct_to_breakout,
        pct_to_sale=pct_to_sale,
        entry_trigger=entry_trigger,
        exit_trigger=exit_trigger,
        status=status,
        pct_change_from_yesterday=pct_change_from_yesterday,
        risk_per_share=risk_per_share,
        position_size=position_size,
        capital_required=capital_required,
    )


# -------------------------------
# Run full watchlist
# -------------------------------

def run_watchlist(watchlist: Dict[str, str]) -> tuple[pd.DataFrame, List[tuple[str, str, str]]]:
    results: List[SignalResult] = []
    errors: List[tuple[str, str, str]] = []

    for name, symbol in watchlist.items():
        try:
            results.append(get_signal(name, symbol))
        except Exception as exc:
            errors.append((name, symbol, str(exc)))

    if not results:
        return pd.DataFrame(), errors

    df = pd.DataFrame([r.__dict__ for r in results])

    df = df[
        [
            "name",
            "symbol",
            "close",
            "prior_100_high",
            "prior_50_low",
            "pct_to_breakout",
            "pct_to_sale",
            "pct_change_from_yesterday",
            "risk_per_share",
            "position_size",
            "capital_required",
            "status",
        ]
    ].sort_values(by=["status", "pct_to_breakout"], ascending=[True, True])

    return df, errors


# -------------------------------
# Portfolio
# -------------------------------

def load_portfolio() -> pd.DataFrame:
    try:
        df = pd.read_csv(PORTFOLIO_FILE)
    except FileNotFoundError:
        return pd.DataFrame()

    required = {"symbol", "name", "entry_price", "position_size", "entry_date", "stop_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"portfolio.csv is missing columns: {sorted(missing)}")

    df = df.copy()
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce").round(PRICE_DECIMALS)
    df["position_size"] = pd.to_numeric(df["position_size"], errors="coerce")
    df["stop_price"] = pd.to_numeric(df["stop_price"], errors="coerce").round(PRICE_DECIMALS)
    df["entry_date"] = df["entry_date"].astype(str)

    df = df.dropna(subset=["symbol", "name", "entry_price", "position_size", "stop_price"])

    return df


def calculate_portfolio_metrics(portfolio: pd.DataFrame, signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if portfolio.empty or signals.empty:
        return pd.DataFrame(), portfolio.copy()

    current_prices = signals[
        ["symbol", "name", "close", "prior_100_high", "prior_50_low", "status"]
    ].copy()

    merged = portfolio.merge(current_prices, on=["symbol", "name"], how="left")

    if merged.empty:
        return pd.DataFrame(), portfolio.copy()

    merged["entry_price_display"] = pd.to_numeric(merged["entry_price"], errors="coerce").round(PRICE_DECIMALS)
    merged["current_price_display"] = pd.to_numeric(merged["close"], errors="coerce").round(PRICE_DECIMALS)
    merged["position_size"] = pd.to_numeric(merged["position_size"], errors="coerce")
    merged["stop_price_display"] = pd.to_numeric(merged["stop_price"], errors="coerce").round(PRICE_DECIMALS)
    merged["today_d50_display"] = pd.to_numeric(merged["prior_50_low"], errors="coerce").round(PRICE_DECIMALS)
    merged["today_d100_display"] = pd.to_numeric(merged["prior_100_high"], errors="coerce").round(PRICE_DECIMALS)

    merged["updated_stop_price_display"] = merged.apply(
        lambda row: normalise_price(max(float(row["stop_price_display"]), float(row["today_d50_display"])))
        if pd.notna(row["stop_price_display"]) and pd.notna(row["today_d50_display"])
        else row["stop_price_display"],
        axis=1,
    )

    merged["stop_moved"] = merged.apply(
        lambda row: "RAISE STOP"
        if pd.notna(row["updated_stop_price_display"])
        and pd.notna(row["stop_price_display"])
        and float(row["updated_stop_price_display"]) > float(row["stop_price_display"])
        else "",
        axis=1,
    )

    merged["exit_signal"] = merged.apply(
        lambda row: "SELL"
        if pd.notna(row["current_price_display"])
        and pd.notna(row["updated_stop_price_display"])
        and float(row["current_price_display"]) < float(row["updated_stop_price_display"])
        else "",
        axis=1,
    )

    # Score % for 100/50 live positions:
    # (Current Price - Prior 100 High) / Prior 100 High * 100
    merged["score_pct_raw"] = merged.apply(
        lambda row: ((float(row["current_price_display"]) - float(row["today_d100_display"])) / float(row["today_d100_display"])) * 100
        if pd.notna(row["current_price_display"]) and pd.notna(row["today_d100_display"]) and float(row["today_d100_display"]) != 0
        else float("nan"),
        axis=1,
    )

    # Highest score = rank 1
    merged["rank"] = merged["score_pct_raw"].rank(ascending=False, method="dense")

    merged["entry_price_calc"] = merged.apply(
        lambda row: convert_price_for_cash_calcs(row["name"], float(row["entry_price_display"])),
        axis=1,
    )

    merged["current_price_calc"] = merged.apply(
        lambda row: convert_price_for_cash_calcs(row["name"], float(row["current_price_display"]))
        if pd.notna(row["current_price_display"]) else float("nan"),
        axis=1,
    )

    merged["stop_price_calc"] = merged.apply(
        lambda row: convert_price_for_cash_calcs(row["name"], float(row["updated_stop_price_display"]))
        if pd.notna(row["updated_stop_price_display"]) else float("nan"),
        axis=1,
    )

    merged["pnl_per_share"] = merged["current_price_calc"] - merged["entry_price_calc"]
    merged["pnl_total"] = merged["pnl_per_share"] * merged["position_size"]
    merged["pnl_pct"] = (merged["pnl_per_share"] / merged["entry_price_calc"]) * 100

    merged["entry_value"] = merged["entry_price_calc"] * merged["position_size"]
    merged["current_value"] = merged["current_price_calc"] * merged["position_size"]
    merged["stop_value"] = merged["stop_price_calc"] * merged["position_size"]

    value_cols = ["entry_value", "current_value", "stop_value", "pnl_total"]
    merged[value_cols] = merged[value_cols].round(PRICE_DECIMALS)
    merged["score_pct_raw"] = merged["score_pct_raw"].round(2)

    updated_portfolio = merged[
        ["symbol", "name", "entry_price", "position_size", "entry_date", "updated_stop_price_display"]
    ].rename(columns={"updated_stop_price_display": "stop_price"}).copy()

    portfolio_metrics = merged[
        [
            "symbol",
            "name",
            "entry_date",
            "entry_price_display",
            "current_price_display",
            "position_size",
            "stop_price_display",
            "today_d50_display",
            "updated_stop_price_display",
            "score_pct_raw",
            "rank",
            "pnl_total",
            "pnl_pct",
            "stop_moved",
            "exit_signal",
        ]
    ].copy()

    return portfolio_metrics, updated_portfolio


def save_updated_portfolio(updated_portfolio: pd.DataFrame) -> None:
    if updated_portfolio.empty:
        print("No portfolio data to save.")
        return

    output_df = updated_portfolio.copy()

    output_df["entry_price"] = pd.to_numeric(output_df["entry_price"], errors="coerce").round(PRICE_DECIMALS)
    output_df["position_size"] = pd.to_numeric(output_df["position_size"], errors="coerce").astype(int)
    output_df["stop_price"] = pd.to_numeric(output_df["stop_price"], errors="coerce").round(PRICE_DECIMALS)

    output_df.to_csv(UPDATED_PORTFOLIO_FILE, index=False, float_format=f"%.{PRICE_DECIMALS}f")

    print(f"\nPortfolio saved to {UPDATED_PORTFOLIO_FILE}")
    print(output_df.to_string(index=False))


# -------------------------------
# Formatting helpers
# -------------------------------

def format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    formatted = df.copy()

    for col in ["close", "prior_100_high", "prior_50_low"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(
                lambda x: f"{float(x):,.2f}" if pd.notna(x) else ""
            )

    for col in ["risk_per_share", "capital_required"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(
                lambda x: f"{float(x):,.2f}" if pd.notna(x) else ""
            )

    for col in ["pct_to_breakout", "pct_to_sale", "pct_change_from_yesterday"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(
                lambda x: f"{float(x):.2f}%" if pd.notna(x) else ""
            )

    if "position_size" in formatted.columns:
        formatted["position_size"] = formatted["position_size"].map(
            lambda x: f"{int(x):,}" if pd.notna(x) else ""
        )

    return formatted


def format_portfolio_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    formatted = df.copy()

    for col in [
        "entry_price_display",
        "current_price_display",
        "stop_price_display",
        "today_d50_display",
        "updated_stop_price_display",
    ]:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(
                lambda x: f"{float(x):,.2f}" if pd.notna(x) else ""
            )

    for col in ["entry_value", "current_value", "stop_value", "pnl_total"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(
                lambda x: f"£{float(x):,.2f}" if pd.notna(x) else ""
            )

    if "pnl_pct" in formatted.columns:
        formatted["pnl_pct"] = formatted["pnl_pct"].map(
            lambda x: f"{float(x):.2f}%" if pd.notna(x) else ""
        )

    if "score_pct_raw" in formatted.columns:
        formatted["score_pct_raw"] = formatted["score_pct_raw"].map(
            lambda x: f"{float(x):.2f}%" if pd.notna(x) else ""
        )

    if "rank" in formatted.columns:
        formatted["rank"] = formatted["rank"].map(
            lambda x: f"{int(float(x))}" if pd.notna(x) else ""
        )

    if "position_size" in formatted.columns:
        formatted["position_size"] = formatted["position_size"].map(
            lambda x: f"{int(x):,}" if pd.notna(x) else ""
        )

    return formatted


def build_summary_html(
    df: pd.DataFrame,
    errors: List[tuple[str, str, str]],
    portfolio_df: pd.DataFrame
) -> str:
    buy_count = 0
    sell_count = 0
    hold_count = 0

    if not df.empty:
        buy_count = int((df["status"] == "BUY").sum())
        sell_count = int((df["status"] == "SELL").sum())
        hold_count = int((df["status"] == "HOLD").sum())

    error_count = len(errors)

    total_current_value = 0.0
    total_pnl = 0.0
    open_positions = 0
    stop_raise_count = 0
    exit_count = 0

    if not portfolio_df.empty:
        portfolio_value_df = portfolio_df.copy()

        portfolio_value_df["current_price_display"] = pd.to_numeric(
            portfolio_value_df["current_price_display"], errors="coerce"
        )
        portfolio_value_df["position_size"] = pd.to_numeric(
            portfolio_value_df["position_size"], errors="coerce"
        )

        portfolio_value_df["current_value_calc"] = portfolio_value_df.apply(
            lambda row: convert_price_for_cash_calcs(row["name"], float(row["current_price_display"])) * float(row["position_size"])
            if pd.notna(row["current_price_display"]) and pd.notna(row["position_size"])
            else 0.0,
            axis=1,
        )

        total_current_value = float(portfolio_value_df["current_value_calc"].sum())
        total_pnl = float(portfolio_df["pnl_total"].sum())
        open_positions = len(portfolio_df)
        stop_raise_count = int((portfolio_df["stop_moved"] == "RAISE STOP").sum())
        exit_count = int((portfolio_df["exit_signal"] == "SELL").sum())

    pnl_bg = "#d4edda" if total_pnl >= 0 else "#f8d7da"
    pnl_color = "#155724" if total_pnl >= 0 else "#721c24"

    return f"""
    <div style="margin-bottom: 20px;">
        <div style="display: inline-block; margin: 0 12px 12px 0; padding: 12px 16px; background: #d4edda; color: #155724; border: 1px solid #c3e6cb; font-weight: bold;">
            BUY: {buy_count}
        </div>
        <div style="display: inline-block; margin: 0 12px 12px 0; padding: 12px 16px; background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; font-weight: bold;">
            SELL: {sell_count}
        </div>
        <div style="display: inline-block; margin: 0 12px 12px 0; padding: 12px 16px; background: #fff3cd; color: #856404; border: 1px solid #ffeeba; font-weight: bold;">
            HOLD: {hold_count}
        </div>
        <div style="display: inline-block; margin: 0 12px 12px 0; padding: 12px 16px; background: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; font-weight: bold;">
            ERRORS: {error_count}
        </div>
        <div style="display: inline-block; margin: 0 12px 12px 0; padding: 12px 16px; background: #e9ecef; color: #212529; border: 1px solid #ced4da; font-weight: bold;">
            RISK / TRADE: £{RISK_PER_TRADE:,.2f}
        </div>
        <div style="display: inline-block; margin: 0 12px 12px 0; padding: 12px 16px; background: #e9ecef; color: #212529; border: 1px solid #ced4da; font-weight: bold;">
            OPEN POSITIONS: {open_positions}
        </div>
        <div style="display: inline-block; margin: 0 12px 12px 0; padding: 12px 16px; background: #d4edda; color: #155724; border: 1px solid #c3e6cb; font-weight: bold;">
            STOPS RAISED: {stop_raise_count}
        </div>
        <div style="display: inline-block; margin: 0 12px 12px 0; padding: 12px 16px; background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; font-weight: bold;">
            PORTFOLIO SELLS: {exit_count}
        </div>
        <div style="display: inline-block; margin: 0 12px 12px 0; padding: 12px 16px; background: #e9ecef; color: #212529; border: 1px solid #ced4da; font-weight: bold;">
            PORTFOLIO VALUE: £{total_current_value:,.2f}
        </div>
        <div style="display: inline-block; margin: 0 12px 12px 0; padding: 12px 16px; background: {pnl_bg}; color: {pnl_color}; border: 1px solid #ced4da; font-weight: bold;">
            TOTAL P&amp;L: £{total_pnl:,.2f}
        </div>
    </div>
    """


def build_actionable_html(df: pd.DataFrame) -> str:
    if df.empty:
        return """
        <p style="margin: 0 0 24px 0;">No signals returned.</p>
        """

    actionable = df[df["status"].isin(["BUY", "SELL"])].copy()

    if actionable.empty:
        return """
        <p style="margin: 0 0 24px 0;">No actionable BUY or SELL signals today.</p>
        """

    actionable = format_for_display(actionable)

    rows = []
    for _, row in actionable.iterrows():
        status = row["status"]

        if status == "BUY":
            status_bg = "#d4edda"
            status_color = "#155724"
        else:
            status_bg = "#f8d7da"
            status_color = "#721c24"

        rows.append(f"""
        <tr>
            <td style="padding:10px; border:1px solid #ddd;">{row['name']}</td>
            <td style="padding:10px; border:1px solid #ddd;">{row['symbol']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['close']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['prior_50_low']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['risk_per_share']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['position_size']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['capital_required']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right; background:{status_bg}; color:{status_color}; font-weight:bold;">
                {status}
            </td>
        </tr>
        """)

    return f"""
    <h3 style="margin: 0 0 12px 0;">Actionable Signals</h3>
    <table style="border-collapse: collapse; width: 100%; font-size: 14px; margin-bottom: 24px;">
        <thead>
            <tr style="background: #060B69; color: #ffffff;">
                <th style="padding:10px; border:1px solid #ddd; text-align:left;">Name</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:left;">Symbol</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Close</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Initial / Current D50</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Risk / Share</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Position Size</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Capital Required</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Status</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def build_portfolio_html(portfolio_df: pd.DataFrame) -> str:
    if portfolio_df.empty:
        return """
        <h3 style="margin: 0 0 12px 0;">Portfolio</h3>
        <p style="margin: 0 0 24px 0;">No active positions in portfolio.csv.</p>
        """

    formatted = format_portfolio_for_display(portfolio_df)

    rows = []
    for idx, row in formatted.iterrows():
        raw_pnl = float(portfolio_df.loc[idx, "pnl_total"])
        pnl_bg = "#d4edda" if raw_pnl >= 0 else "#f8d7da"
        pnl_color = "#155724" if raw_pnl >= 0 else "#721c24"

        stop_moved = row["stop_moved"]
        stop_bg = "#d4edda" if stop_moved == "RAISE STOP" else "#ffffff"
        stop_color = "#155724" if stop_moved == "RAISE STOP" else "#222222"

        exit_signal = row["exit_signal"]
        exit_bg = "#f8d7da" if exit_signal == "SELL" else "#ffffff"
        exit_color = "#721c24" if exit_signal == "SELL" else "#222222"

        rows.append(f"""
        <tr>
            <td style="padding:10px; border:1px solid #ddd;">{row['name']}</td>
            <td style="padding:10px; border:1px solid #ddd;">{row['symbol']}</td>
            <td style="padding:10px; border:1px solid #ddd;">{row['entry_date']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['entry_price_display']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['current_price_display']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['position_size']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['stop_price_display']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['today_d50_display']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['updated_stop_price_display']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['score_pct_raw']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['rank']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right; background:{pnl_bg}; color:{pnl_color}; font-weight:bold;">{row['pnl_total']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right; background:{pnl_bg}; color:{pnl_color}; font-weight:bold;">{row['pnl_pct']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:center; background:{stop_bg}; color:{stop_color}; font-weight:bold;">{stop_moved}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:center; background:{exit_bg}; color:{exit_color}; font-weight:bold;">{exit_signal}</td>
        </tr>
        """)

    return f"""
    <h3 style="margin: 0 0 12px 0;">Portfolio</h3>
    <table style="border-collapse: collapse; width: 100%; font-size: 14px; margin-bottom: 24px;">
        <thead>
            <tr style="background: #060B69; color: #ffffff;">
                <th style="padding:10px; border:1px solid #ddd; text-align:left;">Name</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:left;">Symbol</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:left;">Entry Date</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Entry Price</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Current Price</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Size</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Stored Stop</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Today D50</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">New Stop</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Score %</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Rank</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">P&amp;L</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">P&amp;L %</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:center;">Stop Action</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:center;">Exit Signal</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def build_full_table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return """
        <h3 style="margin: 0 0 12px 0;">Full Watchlist</h3>
        <p>No data returned.</p>
        """

    formatted = format_for_display(df)

    rows = []
    for _, row in formatted.iterrows():
        status = row["status"]

        if status == "BUY":
            status_bg = "#d4edda"
            status_color = "#155724"
        elif status == "SELL":
            status_bg = "#f8d7da"
            status_color = "#721c24"
        else:
            status_bg = "#fff3cd"
            status_color = "#856404"

        rows.append(f"""
        <tr>
            <td style="padding:10px; border:1px solid #ddd;">{row['name']}</td>
            <td style="padding:10px; border:1px solid #ddd;">{row['symbol']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['close']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['prior_100_high']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['prior_50_low']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['pct_to_breakout']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['pct_to_sale']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['pct_change_from_yesterday']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['risk_per_share']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['position_size']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['capital_required']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right; background:{status_bg}; color:{status_color}; font-weight:bold;">
                {status}
            </td>
        </tr>
        """)

    return f"""
    <h3 style="margin: 0 0 12px 0;">Full Watchlist</h3>
    <table style="border-collapse: collapse; width: 100%; font-size: 14px;">
        <thead>
            <tr style="background: #060B69; color: #ffffff;">
                <th style="padding:10px; border:1px solid #ddd; text-align:left;">Name</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:left;">Symbol</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Close</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Prior 100 High</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Prior 50 Low</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">% to Breakout</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">% to Sale</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">% Change</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Risk / Share</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Position Size</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Capital Required</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Status</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def build_errors_html(errors: List[tuple[str, str, str]]) -> str:
    if not errors:
        return ""

    items = []
    for name, symbol, msg in errors:
        items.append(f"<li><strong>{name}</strong> ({symbol}): {msg}</li>")

    return f"""
    <h3 style="margin: 24px 0 12px 0;">Errors</h3>
    <ul style="margin: 0; padding-left: 20px;">
        {''.join(items)}
    </ul>
    """


def build_html_email(
    df: pd.DataFrame,
    errors: List[tuple[str, str, str]],
    portfolio_df: pd.DataFrame
) -> str:
    summary_html = build_summary_html(df, errors, portfolio_df)
    actionable_html = build_actionable_html(df)
    portfolio_html = build_portfolio_html(portfolio_df)
    full_table_html = build_full_table_html(df)
    errors_html = build_errors_html(errors)

    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; color: #222; margin: 0; padding: 24px; background: #f7f7f7;">
        <div style="max-width: 1600px; margin: 0 auto; background: #ffffff; padding: 24px; border: 1px solid #e5e5e5;">
            <h2 style="margin-top: 0;">Daily Market Check</h2>
            <p style="margin: 0 0 18px 0;">Donchian 100 / 50 watchlist scan with position sizing, trailing stop management and portfolio tracking.</p>

            {summary_html}
            {actionable_html}
            {portfolio_html}
            {full_table_html}
            {errors_html}
        </div>
    </body>
    </html>
    """


# -------------------------------
# Main execution
# -------------------------------

def main() -> None:
    signals_df, errors = run_watchlist(WATCHLIST)
    portfolio_input_df = load_portfolio()
    portfolio_metrics_df, updated_portfolio_df = calculate_portfolio_metrics(portfolio_input_df, signals_df)

    print("Daily Market Check")
    print("=" * 120)

    if signals_df.empty:
        print("No signals returned.")
    else:
        print("\nSignals")
        print(format_for_display(signals_df).to_string(index=False))

    if not portfolio_metrics_df.empty:
        print("\nPortfolio")
        print(format_portfolio_for_display(portfolio_metrics_df).to_string(index=False))

        raised = portfolio_metrics_df[portfolio_metrics_df["stop_moved"] == "RAISE STOP"]
        if not raised.empty:
            print("\n=== STOPS RAISED ===")
            print(raised[["name", "symbol", "stop_price_display", "updated_stop_price_display"]].to_string(index=False))
        else:
            print("\nNo stops raised today.")

        print("\n=== UPDATED PORTFOLIO (TO BE SAVED) ===")
        print(updated_portfolio_df.to_string(index=False))

        save_updated_portfolio(updated_portfolio_df)
        print(f"\nUpdated trailing stops written to {UPDATED_PORTFOLIO_FILE}")
    else:
        print("\nPortfolio")
        print("No active positions in portfolio.csv.")

    if errors:
        print("\nErrors:")
        for name, symbol, msg in errors:
            print(f"- {name} ({symbol}): {msg}")

    html = build_html_email(signals_df, errors, portfolio_metrics_df)

    with open("market_email.html", "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()