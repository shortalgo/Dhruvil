# -*- coding: utf-8 -*-
"""
Rhino + Calendar Strategy Implementation
----------------------------------------

This module implements the monthly Rhino (PUT-based) and Calendar (CALL-based) option strategy
with automatic expiry rolling, VIX-regime-based adjustments, delta monitoring, and
data-driven daily expiry alignment (no fixed 30-day assumption).

Entry Logic:
    • Enter on the first Monday of each month at 09:30.
    • Up to 2 Rhinos and 1 Calendar allowed concurrently.

Structures:
    Rhino (PUT legs only)
        leg1 = long put  (~0.50Δ)
        leg2 = short put (~0.35 - 0.40Δ)
        leg3 = long put  (~0.18Δ)
        → 24 lots each long leg (leg1, leg3)
        → 48 lots for short leg (leg2)

    Calendar (CALL legs only)
        Short call on same expiry (~0.40Δ)
        Long call on next expiry (~0.40Δ)
        → 32 lots each leg

Lot size (per contract): 75

Volatility Regime:
    - VIX_Percentile_100D < 75 → Low-vol regime (additive behavior)
    - VIX_Percentile_100D ≥ 75 → High-vol regime (defensive behavior)
    - Rules checked per leg per Rhino and per Calendar.
    - Any condition must persist for 3 consecutive 5-minute checks before acting.
      If it breaks during the 15-minute window, reset its counter.

Universal Exit:
    - Exit all positions if any leg’s DTE < 15 days.

Option Chain Scope:
    - Evaluate only existing positions for delta checks.
    - When adjustment required, rescan from ATM −10 strikes to +15 strikes
      to find replacement legs (using freshly computed deltas).

Expiry Roll Logic (DATA-DRIVEN DAILY ALIGNMENT):
    option_data   → current-month expiry
    option_data1  → next-month expiry
    option_data2  → far-month expiry

    Every trading day, for each open position:
        1) Read mapped_days for that date to get MonthlyExpiry / Next Monthly Expiry / 3rd Monthly Expiry.
        2) If position.expiry == mapped MonthlyExpiry  → quote from option_data
           If position.expiry == mapped Next Monthly  → quote from option_data1
           If position.expiry == mapped 3rd Monthly   → quote from option_data2
        3) If none match (e.g., month rolled), SHIFT the file tier used for that position so it
           references the correct dataset for its stored expiry. Positions remain intact; only data source shifts.

    This guarantees continuous alignment without assuming any fixed ~30-day interval.

Constants:
    LOT_SIZE = 75
    RHINO_LONG_LOTS = 24
    RHINO_SHORT_LOTS = 48
    CALENDAR_LOTS    = 32
    RF_RATE          = 0.067
    PERSIST_BARS     = 3   # 3 × 5-minute bars = 15 min
    UNIVERSAL_EXIT_DTE = 15

Inputs (DataFrames expected):
    mapped_days: Date, MonthlyExpiry, Next Monthly Expiry, 3rd Monthly Expiry
    vix_df:      Date, Open, High, Low, Close, VIX_Percentile_100D (daily)
    index_df:    Date, Time, Spot (minute)
    option_data:    monthly  (minute; has Date, Time, Type, StrikePrice, ExpiryDate, Open)
    option_data1:   next     (minute; same columns)
    option_data2:   far      (minute; same columns)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, time as dt_time, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
import os
from helpers import pull_options_data_d, pull_index_data
from functools import lru_cache
from time import perf_counter
import re
from collections import defaultdict

from multiprocessing import Pool, cpu_count, get_start_method
import itertools

DELTA_POOL = None  # set in __main__; used inside pick_leg_by_delta

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*concatenation with empty or all-NA entries.*"
)

# --- Configuration & Constants ---
stock = "NIFTY"
roundoff = 50 if stock == 'NIFTY' else (100 if stock == 'SENSEX' else 50)
LOT_SIZE = 75 if stock == 'NIFTY' else (20 if stock == 'SENSEX' else 40)
brokerage = 4.5 if stock == 'NIFTY' else (3 if stock == 'SENSEX' else 3)

root_path = rf"/home/newberry2/dhruvil/rhino/"
option_data_path = rf"/home/newberry2/dhruvil/Current_Expiry_Month_OI_OHLC/"
option_data_path1 = rf"/home/newberry2/dhruvil/Next_Monthly_OI_OHLC/"
option_data_path2 = rf"/home/newberry2/dhruvil/Far_Month_Expiry_OI_OHLC/"

expiry_file_path = rf"{root_path}/nifty_expiry_dates_main.xlsx"
output_folder_path = rf'{root_path}/Trade_Sheets/split/'
filter_df_path = rf"{root_path}/Filter_Sheets/"
txt_file_path = rf'{root_path}/new_done.txt'
runlog_path = rf"{root_path}/run_times.log"  # text log of timings


os.makedirs(output_folder_path, exist_ok=True)
os.makedirs(filter_df_path, exist_ok=True)
open(txt_file_path, 'a').close()


# ===================== CONFIG =====================

date_ranges = [
    ('2024-01-01', '2025-08-22'),
]

year = "main"

LOT_SIZE = 75
RHINO_LONG_LOTS = 24
RHINO_SHORT_LOTS = 48
CALENDAR_LOTS = 32
RF_RATE = 0.067
PERSIST_BARS = 3
UNIVERSAL_EXIT_DTE = 15  # days
ENTRY_DAY = 0            # Monday
ENTRY_TIME_STR = "09:30"
DELTA_TARGETS_RHINO = {
    "leg1_long": 0.50,   # ~0.50Δ long put
    "leg2_short": 0.35,  # <=0.35–0.40Δ short put
    "leg3_long": 0.18    # <=0.18Δ long put
}
DELTA_TARGET_CAL = 0.40   # ~0.40Δ call for both legs
# STRIKE_WIN_BELOW = 25
# STRIKE_WIN_ABOVE = 15
# ---- Day marker times (India market) ----
DAY_OPEN_TIME = "09:30"
DAY_EOD_TIME  = "15:15"

PUT_WIN_BELOW  = 45   # wider below ATM (puts)
PUT_WIN_ABOVE  = 25   # narrow above ATM (puts)
CALL_WIN_BELOW = 25    # narrow below ATM (calls)
CALL_WIN_ABOVE = 25   # wider above ATM (calls)



# ================= Option math helpers =================
def black_scholes_price(S, K, T, r, sigma, opt_type):
    if sigma is None or sigma <= 0 or T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r*T)*norm.cdf(d2)
    else:
        return K * np.exp(-r*T)*norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_volatility(price, S, K, T, r, opt_type):
    try:
        return brentq(lambda sig: black_scholes_price(S,K,T,r,sig,opt_type) - price,
                      1e-4, 3.0, maxiter=150)
    except Exception:
        return None

def black_scholes_greeks(S, K, T, r, sigma, opt_type):
    if sigma is None or sigma <= 0 or T <= 0:
        return {'Delta': None, 'Gamma': None, 'Vega': None}
    d1 = (np.log(S / K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    if opt_type.lower() == 'call':
        delta = norm.cdf(d1)
    else:
        # put delta is -Phi(-d1) which equals Phi(d1)-1; we keep explicit sign
        delta = -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return {'Delta': delta, 'Gamma': gamma, 'Vega': vega}

def _iv_delta_worker(args):
    """Pure worker for multiprocessing: returns (idx, T, Delta or np.nan)."""
    i, px, spot, K, exp_ts, ref_dt, r, optype = args
    # compute T using your existing helper
    T, _ = calculate_time_to_expiry(exp_ts, ref_dt)
    if T <= 0:
        return (i, T, np.nan)
    iv = iv_cached(px, spot, K, T, r, optype)
    d  = delta_cached(spot, K, T, r, iv, optype)
    return (i, T, (np.nan if d is None else d))


def calculate_time_to_expiry(expiry_date, reference_datetime):
    """Trading-day fraction using India hours 09:15–15:30; returns T in years and DTE (days)."""
    # normalize both inputs to pd.Timestamp (handles str, numpy.datetime64, datetime, etc.)
    try:
        expiry_ts = pd.to_datetime(expiry_date)
    except Exception:
        expiry_ts = pd.to_datetime(str(expiry_date), errors='coerce')

    try:
        ref_ts = pd.to_datetime(reference_datetime)
    except Exception:
        ref_ts = pd.to_datetime(str(reference_datetime), errors='coerce')

    # guard NaT
    if pd.isna(expiry_ts) or pd.isna(ref_ts):
        return 0.0, 0

    market_open = dt_time(9, 15)
    market_close = dt_time(15, 30)
    full_day_minutes = (market_close.hour*60 + market_close.minute) - (market_open.hour*60 + market_open.minute)

    # minutes left today (clamped to trading session)
    minutes_since_open = (ref_ts.hour * 60 + ref_ts.minute) - (market_open.hour * 60 + market_open.minute)
    minutes_left_today = max(0, full_day_minutes - max(0, minutes_since_open))
    fraction_today = minutes_left_today / full_day_minutes if minutes_left_today > 0 else 0

    total_days = (expiry_ts.date() - ref_ts.date()).days
    full_days_after = max(0, total_days - 1)
    trading_days = fraction_today + full_days_after

    T_years = round(trading_days / 365, 6)
    dte_days = int(full_days_after + (1 if fraction_today > 0 else 0))
    return T_years, dte_days

# ---- Snapshot & tier caches (logic-preserving) ----


_SNAP_ROW_CACHE = {}          # (date, time, expiry, strike, kind) -> {'price','delta','Expiry','strike'}
_RESOLVE_TIER_CACHE = {}      # (date, expiry) -> 'monthly'|'next'|'far'

def _key_norm(date_, time_str, expiry, strike, kind):
    return (pd.to_datetime(date_).date(),
            str(time_str),
            pd.to_datetime(expiry).date(),
            float(strike),
            str(kind).lower())

def resolve_chain_for_expiry_cached(date_, mapped_days_row, expiry_target,
                                    option_data, option_data1, option_data2):
    """Memoized wrapper around resolve_chain_for_expiry; returns (df, tier)."""
    k = (pd.to_datetime(date_).date(), pd.to_datetime(expiry_target).date())
    if k not in _RESOLVE_TIER_CACHE:
        _, tier = resolve_chain_for_expiry(date_, mapped_days_row, expiry_target,
                                           option_data, option_data1, option_data2)
        _RESOLVE_TIER_CACHE[k] = tier
    tier = _RESOLVE_TIER_CACHE[k]
    df = {'monthly': option_data, 'next': option_data1, 'far': option_data2}[tier]
    return df, tier

# change signature
def snapshot_row(chain_minute_df, ref_dt, expiry, strike, kind, spot, rf=RF_RATE):
    key = (_key_norm(ref_dt.date(), ref_dt.strftime("%H:%M"), expiry, strike, kind))
    if key in _SNAP_ROW_CACHE:
        return _SNAP_ROW_CACHE[key]
    if chain_minute_df.empty:
        return {}

    exp_date = pd.to_datetime(expiry).date()
    row = chain_minute_df[
        (chain_minute_df['StrikePrice'] == float(strike)) &
        (chain_minute_df['Type'].str.lower() == str(kind).lower()) &
        (pd.to_datetime(chain_minute_df['ExpiryDate']).dt.date == exp_date)
    ]
    if row.empty:
        return {}

    r0 = row.iloc[0]
    T, _ = calculate_time_to_expiry(pd.to_datetime(r0['ExpiryDate']), ref_dt)

    # STRICT: only the minute’s Open; if not usable → missing
    price = float(r0.get('Open', np.nan))
    if (not pd.notna(price)) or price <= 0:
        return {}

    iv = iv_cached(price, float(spot), float(strike), T, rf, str(kind))
    d  = delta_cached(float(spot), float(strike), T, rf, iv, str(kind))

    out = {
        'strike': float(strike),
        'delta': None if d is None else float(d),
        'price': price,                           # exact minute price only
        'expiry': pd.to_datetime(r0['ExpiryDate']).date()
    }
    _SNAP_ROW_CACHE[key] = out
    return out


def _clear_snapshot_caches():
    _SNAP_ROW_CACHE.clear()
    _RESOLVE_TIER_CACHE.clear()

# Pnl Tracking
def _timestamp(date, tstr):
    return pd.to_datetime(str(pd.to_datetime(date).date()) + " " + str(tstr))

def _role_dir_map(stype):
    # +1 for long, -1 for short
    return (
        {'long1': +1, 'short': -1, 'long2': +1} if str(stype).lower() == 'rhino'
        else {'cal_short': -1, 'cal_long': +1}
    )

def _leg_roles(stype):
    return ('long1','short','long2') if str(stype).lower() == 'rhino' else ('cal_short','cal_long')

def _md_row_for_date(mapped_days, d):
    row = mapped_days[mapped_days['Date'].dt.date == d]
    return None if row.empty else row.iloc[0]


# Tracks which slots are occupied per PositionID

def make_pid(expiry_ts: pd.Timestamp) -> str:
    return f"PID-{pd.to_datetime(expiry_ts).strftime('%Y%m%d')}"

position_groups = {}  # { pid: {"R1": sid_or_None, "R2": sid_or_None, "CAL": sid_or_None} }

def _min(x: float) -> float:
    return x / 60.0

def build_rhino_lifecycle(fills_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-leg lifecycle (one row per leg) for RHINO.
    Output has a plain PID (no '|R1'), sorted later by PID + Entry Date/Time.
    """
    df = fills_df.copy()
    df = df[df['Type'].str.lower() == 'rhino']  # only rhino structures
    df['StructureKey'] = df['StructureID'] + "|" + df['Slot']   # internal only
    df = df.sort_values(['Date','Time'], kind='mergesort')

    leg_meta  = {'long1': {'lots': RHINO_LONG_LOTS,  'direction': +1},
                 'short': {'lots': RHINO_SHORT_LOTS, 'direction': -1},
                 'long2': {'lots': RHINO_LONG_LOTS,  'direction': +1}}
    leg_order = ['long1','short','long2']

    out_rows = []

    for skey, g in df.groupby('StructureKey'):
        g = g.sort_values(['Date','Time'], kind='mergesort')
        instance = 0
        open_row = None

        # plain PID (drop slot)
        pid_plain = skey.split('|', 1)[0]

        for _, row in g.iterrows():
            action = str(row['Action'])

            if action == 'OPEN' and open_row is None:
                open_row = row
                instance += 1
                continue

            if (action.startswith('EXIT') or action.startswith('UNIVERSAL_EXIT')) and open_row is not None:
                # Month label (optional; keep if useful)
                rhino_month = open_row['StructureID'].replace('PID-', '')
                rhino_month = pd.to_datetime(rhino_month, format='%Y%m%d', errors='coerce')
                rhino_month = "" if pd.isna(rhino_month) else str(pd.Period(rhino_month, freq="M"))

                for leg in leg_order:
                    lots      = leg_meta[leg]['lots']
                    direction = leg_meta[leg]['direction']

                    e_price  = open_row.get(f'{leg}_Price',  np.nan)
                    e_delta  = open_row.get(f'{leg}_Delta',  np.nan)
                    e_strike = open_row.get(f'{leg}_Strike', np.nan)
                    e_exp    = open_row.get(f'{leg}_Expiry', np.nan)

                    x_price  = row.get(f'{leg}_Price',  np.nan)
                    x_delta  = row.get(f'{leg}_Delta',  np.nan)

                    if pd.notna(e_price) and pd.notna(x_price):
                        leg_pnl = (float(x_price) - float(e_price)) * float(direction) * float(lots) * float(LOT_SIZE)
                    else:
                        leg_pnl = np.nan

                    out_rows.append({
                        'PID'         : pid_plain,           # <<< plain PID
                        'Rhino_month' : rhino_month,
                        'Instance'    : instance,

                        'Entry Date'  : open_row['Date'],
                        'Entry Time'  : open_row['Time'],
                        'Quantity'    : lots * LOT_SIZE,
                        'Expirydate'  : e_exp,
                        'Strike'      : e_strike,
                        'Type'        : leg,                # long1/short/long2

                        'Entry Price' : e_price,
                        'Entry Delta' : e_delta,

                        'Exit Date'   : row['Date'],
                        'Exit Time'   : row['Time'],
                        'Exit Price'  : x_price,
                        'Exit Delta'  : x_delta,

                        'P&L'         : leg_pnl,
                    })

                open_row = None  # matched; wait for next instance

    lf = pd.DataFrame(out_rows)
    desired_cols = [
        'PID','Rhino_month','Instance',
        'Entry Date','Entry Time','Quantity','Expirydate','Strike','Type',
        'Entry Price','Entry Delta',
        'Exit Date','Exit Time','Exit Price','Exit Delta',
        'P&L'
    ]
    if not lf.empty:
        lf = lf.reindex(columns=desired_cols)
    return lf

def build_calendar_lifecycle(fills_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-leg lifecycle (one row per leg) for CALENDAR.
    Output has a plain PID (no '|CAL'), sorted later by PID + Entry Date/Time.
    """
    df = fills_df.copy()
    df = df[df['Type'].str.lower() == 'calendar']
    df['StructureKey'] = df['StructureID'] + "|" + df['Slot']   # internal only
    df = df.sort_values(['Date','Time'], kind='mergesort')

    leg_meta  = {'cal_short': {'lots': CALENDAR_LOTS, 'direction': -1},
                 'cal_long' : {'lots': CALENDAR_LOTS, 'direction': +1}}
    leg_order = ['cal_short','cal_long']

    out_rows = []

    for skey, g in df.groupby('StructureKey'):
        g = g.sort_values(['Date','Time'], kind='mergesort')
        instance = 0
        open_row = None

        # plain PID (drop slot)
        pid_plain = skey.split('|', 1)[0]

        for _, row in g.iterrows():
            action = str(row['Action'])

            if action == 'OPEN' and open_row is None:
                open_row = row
                instance += 1
                continue

            if (action.startswith('EXIT') or action.startswith('UNIVERSAL_EXIT')) and open_row is not None:
                cal_month = open_row['StructureID'].replace('PID-', '')
                cal_month = pd.to_datetime(cal_month, format='%Y%m%d', errors='coerce')
                cal_month = "" if pd.isna(cal_month) else str(pd.Period(cal_month, freq="M"))

                for leg in leg_order:
                    lots      = leg_meta[leg]['lots']
                    direction = leg_meta[leg]['direction']

                    e_price  = open_row.get(f'{leg}_Price',  np.nan)
                    e_delta  = open_row.get(f'{leg}_Delta',  np.nan)
                    e_strike = open_row.get(f'{leg}_Strike', np.nan)
                    e_exp    = open_row.get(f'{leg}_Expiry', np.nan)

                    x_price  = row.get(f'{leg}_Price',  np.nan)
                    x_delta  = row.get(f'{leg}_Delta',  np.nan)

                    if pd.notna(e_price) and pd.notna(x_price):
                        leg_pnl = (float(x_price) - float(e_price)) * float(direction) * float(lots) * float(LOT_SIZE)
                    else:
                        leg_pnl = np.nan

                    out_rows.append({
                        'PID'         : pid_plain,          # <<< plain PID
                        'Rhino_month' : cal_month,          # keep same column name
                        'Instance'    : instance,

                        'Entry Date'  : open_row['Date'],
                        'Entry Time'  : open_row['Time'],
                        'Quantity'    : lots * LOT_SIZE,
                        'Expirydate'  : e_exp,
                        'Strike'      : e_strike,
                        'Type'        : leg,                # cal_short / cal_long

                        'Entry Price' : e_price,
                        'Entry Delta' : e_delta,

                        'Exit Date'   : row['Date'],
                        'Exit Time'   : row['Time'],
                        'Exit Price'  : x_price,
                        'Exit Delta'  : x_delta,

                        'P&L'         : leg_pnl,
                    })

                open_row = None

    lf = pd.DataFrame(out_rows)
    desired_cols = [
        'PID','Rhino_month','Instance',
        'Entry Date','Entry Time','Quantity','Expirydate','Strike','Type',
        'Entry Price','Entry Delta',
        'Exit Date','Exit Time','Exit Price','Exit Delta',
        'P&L'
    ]
    if not lf.empty:
        lf = lf.reindex(columns=desired_cols)
    return lf

def _ensure_dt_time_cols(df):
    out = df.copy()
    if 'Date' in out.columns:
        out['Date'] = pd.to_datetime(out['Date'], errors='coerce')
    if 'Time' in out.columns:
        out['Time'] = out['Time'].astype(str).str.slice(0, 5)  # HH:MM
    # Build TS if possible
    if 'Date' in out.columns and 'Time' in out.columns and 'TS' not in out.columns:
        try:
            out['TS'] = pd.to_datetime(out['Date'].dt.date.astype(str) + ' ' + out['Time'],
                                       errors='coerce')
        except Exception:
            pass
    return out

def sort_by_date_time_pid(df):
    """
    Stable sort. Prefer TS if available; otherwise Date→Time.
    Append PID to the sort keys only if that column exists.
    """
    out = _ensure_dt_time_cols(df)

    by = []
    if 'TS' in out.columns and out['TS'].notna().any():
        by.append('TS')
    else:
        if 'Date' in out.columns:
            by.append('Date')
        if 'Time' in out.columns:
            by.append('Time')
    if 'PID' in out.columns:
        by.append('PID')

    # If nothing to sort by, return as-is
    if not by:
        return out.reset_index(drop=True)

    return out.sort_values(by, kind='mergesort').reset_index(drop=True)

def ensure_group(pid: str):
    if pid not in position_groups:
        position_groups[pid] = {"R1": None, "R2": None, "CAL": None}

def first_free_rhino_slot(pid: str):
    ensure_group(pid)
    for slot in ("R1", "R2"):
        if position_groups[pid][slot] is None:
            return slot
    return None

def rhino_slots_count(pid: str):
    ensure_group(pid)
    return sum(1 for s in ("R1","R2") if position_groups[pid][s] is not None)

def calendar_exists(pid: str):
    ensure_group(pid)
    return position_groups[pid]["CAL"] is not None

def count_open_rhinos_for_pid(open_structs, pid: str) -> int:
    return sum(1 for s in open_structs
               if s.stype == "rhino" and s.status == "open" and s.pid == pid)

# --- Debug/toggles (printing only; no logic impact) ---
DEBUG = False
PRINT_CANDIDATES = False          # show per-minute candidate lists
PRINT_CANDIDATE_COUNTS = False    # show candidate strike counts/coverage
PRINT_PNL_SUMMARY = False         # print per-PID PnL summary (final/max/min with times)
SHOW_PNL_SUMMARY = False   # set False to silence the console summary

def dbg(tag, **kw):
    """Tiny debug printer."""
    if not DEBUG:
        return
    parts = [f"{k}={v}" for k, v in kw.items()]
    print(f"[{tag}] " + " | ".join(parts))

def log_time(msg: str):
    """Print to console and append to a log file."""
    print(msg)
    try:
        with open(runlog_path, "a") as f:
            f.write(msg + "\n")
    except Exception as e:
        print(f"[log_time] failed to write: {e}")

def _debug_print_candidates(chain_minute_df, spot, ref_dt):
    """
    Print candidate strikes (with Δ and price) for PUT/CALL windows
    around ATM using your global window constants.
    """
    if not PRINT_CANDIDATES:
        return
    if chain_minute_df is None or chain_minute_df.empty:
        print(f"[CANDIDATES] {ref_dt} | No chain minute rows.")
        return

    # Build a sorted strike list from this minute
    strikes_all = _strike_list_for_minute(chain_minute_df)
    if not strikes_all:
        print(f"[CANDIDATES] {ref_dt} | No strikes in chain minute.")
        return

    atm = get_atm_strike(strikes_all, spot)

    def _dump(optype, lower, upper, label):
        # Window of strikes around ATM using your window constants
        window = set(get_strike_window(strikes_all, spot, lower, upper))
        # Filter rows for this optype and window
        df = chain_minute_df[
            (chain_minute_df['Type'].str.lower() == optype) &
            (chain_minute_df['StrikePrice'].isin(window))
        ].copy()

        rows = []
        for _, r in df.iterrows():
            K = float(r['StrikePrice'])
            price = float(r['Open'])
            T, _ = calculate_time_to_expiry(pd.to_datetime(r['ExpiryDate']), ref_dt)
            iv = iv_cached(price, float(spot), K, T, RF_RATE, optype)
            d  = delta_cached(float(spot), K, T, RF_RATE, iv, optype)
            rows.append((K, d, price))

        rows.sort(key=lambda x: x[0])  # sort by strike
        # Pretty print
        print(
            f"[CANDIDATES] {ref_dt.strftime('%Y-%m-%d %H:%M')} | Spot={float(spot):.2f} | "
            f"ATM={atm} | {label} ({optype.upper()})"
        )

        if not rows:
            print("  (none)")
        else:
            for K, d, px in rows:
                d_str = "NA" if d is None else f"{d:.6f}"
                print(f"  Strike={int(K):<7}  Δ={d_str:<12}  Px={px}")

    # PUT windows
    _dump('put',  PUT_WIN_BELOW,  PUT_WIN_ABOVE,  f"PUT_WIN_BELOW={PUT_WIN_BELOW}, PUT_WIN_ABOVE={PUT_WIN_ABOVE}")
    # CALL windows
    _dump('call', CALL_WIN_BELOW, CALL_WIN_ABOVE, f"CALL_WIN_BELOW={CALL_WIN_BELOW}, CALL_WIN_ABOVE={CALL_WIN_ABOVE}")

def _debug_print_candidate_counts(
    chain_minute_df,
    chain_all_df,
    spot,
    ref_dt,
    label,
    exp_target=None,          # 👈 pass the expiry you're targeting (Timestamp/str/date)
    show_missing=True,        # 👈 print which strikes are missing from this minute
    max_list=60               # 👈 cap how many missing strikes we print
):
    """
    Print ATM, SPOT, and counts of strikes below/at/above ATM for both PUT and CALL windows.

    Improvements:
      - Pins the universe to a specific expiry if exp_target is provided.
      - Normalizes all strikes to ints for robust matching.
      - Optionally prints the exact strikes in the window that are missing at this minute.
    """

    if not PRINT_CANDIDATE_COUNTS:
        return
    
    if chain_all_df is None or chain_all_df.empty:
        print(f"[CANDS-COUNT] {ref_dt:%Y-%m-%d %H:%M} | {label} | (no chain_all rows)")
        return

    day = pd.to_datetime(ref_dt).date()

    day_slice = chain_all_df[chain_all_df['Date'].dt.date == day].copy()
    if exp_target is not None:
        exp_target = pd.to_datetime(exp_target).date()
        day_slice = day_slice[pd.to_datetime(day_slice['ExpiryDate']).dt.date == exp_target]

    # Normalize to int strikes
    day_slice['StrikePrice'] = pd.to_numeric(day_slice['StrikePrice'], errors='coerce')
    strikes_day = sorted({int(k) for k in day_slice['StrikePrice'].dropna().tolist()})
    if not strikes_day:
        print(f"[CANDS-COUNT] {ref_dt:%Y-%m-%d %H:%M} | {label} | (no strikes on day)")
        return

    atm = get_atm_strike(strikes_day, spot)

    def _stats_for(optype, lower, upper):
        window = get_strike_window(strikes_day, spot, lower, upper)
        window = [int(k) for k in window]  # normalize to int

        below = sum(1 for k in window if k < atm)
        at    = sum(1 for k in window if k == atm)
        above = sum(1 for k in window if k > atm)

        if chain_minute_df is not None and not chain_minute_df.empty:
            dfm = chain_minute_df.copy()
            # restrict type
            dfm = dfm[dfm['Type'].str.lower() == optype]
            # restrict expiry if provided
            if exp_target is not None and 'ExpiryDate' in dfm.columns:
                dfm = dfm[pd.to_datetime(dfm['ExpiryDate']).dt.date == exp_target]
            # normalize minute strikes to int
            minute_strikes = {
                int(x) for x in pd.to_numeric(dfm['StrikePrice'], errors='coerce').dropna().tolist()
            }
            present_minute = sum(1 for k in window if k in minute_strikes)
            missing_list = [k for k in window if k not in minute_strikes]
        else:
            minute_strikes = set()
            present_minute = 0
            missing_list = list(window)

        out = {
            "total_in_window": len(window),
            "below": below, "at": at, "above": above,
            "present_minute": present_minute,
            "missing_minute": len(window) - present_minute,
            "window": window,
            "minute_strikes": minute_strikes,
            "missing_list": missing_list
        }
        return out

    put_stats  = _stats_for('put',  PUT_WIN_BELOW,  PUT_WIN_ABOVE)
    call_stats = _stats_for('call', CALL_WIN_BELOW, CALL_WIN_ABOVE)

    print(f"[CANDS-COUNT] {ref_dt:%Y-%m-%d %H:%M} | {label}")
    print(f"  SPOT={spot:.2f} | ATM={atm} | EXP={exp_target if exp_target else 'ALL-IN-TIER'}")

    print(f"  PUT  win (below={PUT_WIN_BELOW}, above={PUT_WIN_ABOVE}): "
          f"total={put_stats['total_in_window']}  below={put_stats['below']}  at={put_stats['at']}  above={put_stats['above']}  "
          f"[present_minute={put_stats['present_minute']}, missing_minute={put_stats['missing_minute']}]")
    if show_missing and put_stats['missing_list']:
        miss = put_stats['missing_list'][:max_list]
        ell  = " …" if len(put_stats['missing_list']) > max_list else ""
        print(f"    PUT missing strikes: {miss}{ell}")

    print(f"  CALL win (below={CALL_WIN_BELOW}, above={CALL_WIN_ABOVE}): "
          f"total={call_stats['total_in_window']} below={call_stats['below']} at={call_stats['at']} above={call_stats['above']} "
          f"[present_minute={call_stats['present_minute']}, missing_minute={call_stats['missing_minute']}]")
    if show_missing and call_stats['missing_list']:
        miss = call_stats['missing_list'][:max_list]
        ell  = " …" if len(call_stats['missing_list']) > max_list else ""
        print(f"    CALL missing strikes: {miss}{ell}")


# ================= Utilities =================
def normalize_option_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Date/Time/Type for CE/PE→call/put, ensure numeric strikes/prices."""
    out = df.copy()
    # Date/Time
    out['Date'] = pd.to_datetime(out['Date'], errors='coerce')
    out = normalize_time_column(out)
    # Expiry
    out['ExpiryDate'] = pd.to_datetime(out['ExpiryDate'], errors='coerce')
    # Strike & prices
    for c in ['StrikePrice','Open','High','Low','Close','OI']:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce')
    # Type normalization
    t = out['Type'].astype(str).str.strip().str.upper()

    # quick diagnostics (optional, during investigations)
    vc_raw = t.value_counts(dropna=False)
    

    # map strictly CE/PE
    mapped = t.map({'CE': 'call', 'PE': 'put'})

    # detect anything unexpected
    bad_mask = mapped.isna()
    if bad_mask.any():
        bad_vals = sorted(t[bad_mask].unique().tolist())
        print("[Type] ❗ unexpected values (will be dropped):", bad_vals)

    # keep only rows that mapped to call/put
    out = out.loc[~bad_mask].copy()
    out['Type'] = mapped
    return out

def get_atm_strike(strikes, spot):
    return min(strikes, key=lambda x: abs(x - spot))

def _clear_strike_list_cache():
    _STRIKE_LIST_CACHE.clear()

# REPLACE the whole function with this:
def get_strike_window(strikes, spot, lower, upper):
    atm = get_atm_strike(strikes, spot)
    sorted_strikes = sorted(strikes)
    idx = sorted_strikes.index(atm)
    start = max(0, idx - lower)
    end = min(len(sorted_strikes), idx + upper + 1)
    return sorted_strikes[start:end]

_STRIKE_LIST_CACHE = {}  # key: id(df) -> sorted strike list

def _strike_list_for_minute(df):
    k = id(df)
    lst = _STRIKE_LIST_CACHE.get(k)
    if lst is None:
        lst = sorted(df['StrikePrice'].unique().tolist())
        _STRIKE_LIST_CACHE[k] = lst
    return lst

def extract_option_chain_slice(option_minute_df, spot, optype=None):
    strikes = sorted(option_minute_df['StrikePrice'].dropna().unique().tolist())  # safer, minute-true
    if optype == 'put':
        window = set(get_strike_window(strikes, spot, lower=PUT_WIN_BELOW, upper=PUT_WIN_ABOVE))
    elif optype == 'call':
        window = set(get_strike_window(strikes, spot, lower=CALL_WIN_BELOW, upper=CALL_WIN_ABOVE))
    else:
        window = set(get_strike_window(strikes, spot, lower=PUT_WIN_BELOW, upper=PUT_WIN_ABOVE))
    # boolean mask is faster than isin on Python set → make a numpy mask
    mask = option_minute_df['StrikePrice'].isin(window)
    return option_minute_df.loc[mask]

def pick_leg_by_delta(slice_df, spot, r, ref_dt, target_delta, optype, direction):
    df = slice_df[slice_df['Type'].str.lower() == optype].copy()
    if df.empty:
        return None

    exp = pd.to_datetime(df['ExpiryDate']).values
    strikes = df['StrikePrice'].values
    px = df['Open'].values

    n = len(df)
    T_arr = np.empty(n, dtype=float)
    D_arr = np.empty(n, dtype=float)
    D_arr[:] = np.nan

    # --- parallel path when pool is available & batch is worthwhile ---
    use_pool = (DELTA_POOL is not None) and (n >= 32)  # threshold avoids overhead on tiny batches
    if use_pool:
        args_iter = (
            (i, float(px[i]), float(spot), float(strikes[i]), exp[i], ref_dt, r, optype)
            for i in range(n)
        )
        # chunksize heuristic keeps workers busy without too much overhead
        chunksize = max(1, n // (4 * 4))
        for i, T, d in DELTA_POOL.imap_unordered(_iv_delta_worker, args_iter, chunksize=chunksize):
            T_arr[i] = T
            if not (d is None or (isinstance(d, float) and np.isnan(d))):
                D_arr[i] = d
    else:
        # --- serial fallback (small batches or no pool) ---
        for i in range(n):
            T, _ = calculate_time_to_expiry(exp[i], ref_dt)
            iv = iv_cached(px[i], spot, strikes[i], T, r, optype)
            d  = delta_cached(spot, strikes[i], T, r, iv, optype)
            T_arr[i] = T
            if d is not None:
                D_arr[i] = d

    df['T'] = T_arr
    df['Delta'] = D_arr
    df = df.dropna(subset=['Delta'])
    df['abs_delta'] = np.abs(df['Delta'])

    if direction == '>=':
        cand = df[df['abs_delta'] >= abs(target_delta)].copy()
    else:
        cand = df[df['abs_delta'] <= abs(target_delta)].copy()
    if cand.empty:
        return None

    cand['diff'] = (cand['abs_delta'] - abs(target_delta)).abs()
    cand = cand.assign(dist2spot=(cand['StrikePrice'] - spot).abs())
    sel = (cand.sort_values(['diff','dist2spot','StrikePrice'],
            ascending=[True, True, (optype=='put')], kind='mergesort').iloc[0])
    return sel


def normalize_time_column(df):
    def safe_parse(t):
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(str(t).strip(), fmt).strftime("%H:%M")
            except:
                continue
        return None
    out = df.copy()
    out['Time'] = out['Time'].apply(safe_parse)
    return out

def is_first_monday_930(date, time_str):
    if time_str != ENTRY_TIME_STR:
        return False
    # first Monday of that month
    rng = pd.date_range(start=date.replace(day=1), end=date, freq='W-MON')
    return (len(rng) > 0) and (date == rng[0].date())

def minute_is_5m(time_str):
    try:
        m = int(time_str[-2:])
        return m % 5 == 0
    except Exception:
        return False

# --- small guards / rounding helpers ---
def any_none(*xs):
    return any(x is None for x in xs)

def _rf(x, n=8):  # round float for cache keys (reduces float-noise for lru_cache)
    return None if x is None else round(float(x), n)

@lru_cache(maxsize=500_000)
def iv_cached(price, S, K, T, r, opt_type):
    # hashable + stable args
    return implied_volatility(_rf(price,6), _rf(S,6), _rf(K,2), _rf(T,8), _rf(r,6), str(opt_type))

@lru_cache(maxsize=500_000)
def delta_cached(S, K, T, r, iv, opt_type):
    if iv is None or iv <= 0 or T <= 0:
        return None
    g = black_scholes_greeks(_rf(S,6), _rf(K,2), _rf(T,8), _rf(r,6), _rf(iv,6), str(opt_type))
    return g.get('Delta', None)


# ================== Position models ==================
@dataclass
class Leg:
    kind: str            # "put" or "call"
    role: str            # "long1"/"short"/"long2" or "cal_short"/"cal_long"
    strike: float
    expiry: pd.Timestamp
    lots: int
    entry_price: float
    direction: int       # +1 long, -1 short

@dataclass
class Structure:
    sid: str
    stype: str           # "rhino" or "calendar"
    opened_at: pd.Timestamp
    expiry: pd.Timestamp
    legs: list = field(default_factory=list)
    status: str = "open"
    counters: dict = field(default_factory=dict)  # keep only once
    pid: str = ""        # PositionID (PID-YYYYMMDD)
    slot: str = ""       # "R1"/"R2"/"CAL"

    def reset_counter(self, key):
        self.counters[key] = 0
    def bump_counter(self, key):
        self.counters[key] = self.counters.get(key, 0) + 1
    def reached_persist(self, key):
        return self.counters.get(key, 0) >= PERSIST_BARS

def structure_pnl(st: Structure, exit_snapshot: dict) -> float:
    """
    Sum PnL across legs for this structure instance:
    PnL_leg = (exit_price - entry_price) * direction * lots * LOT_SIZE
    Uses Leg.entry_price for entry, and exit_snapshot[role]['price'] for exit.
    """
    total = 0.0
    for lg in st.legs:
        if lg.role not in exit_snapshot or 'price' not in exit_snapshot[lg.role]:
            continue
        px_in  = float(lg.entry_price)
        px_out = float(exit_snapshot[lg.role]['price'])
        leg_pnl = (px_out - px_in) * float(lg.direction) * float(lg.lots) * float(LOT_SIZE)
        total += leg_pnl
    return total

# =========== Chain resolution: DAILY expiry alignment ===========
def resolve_chain_for_expiry(date_, mapped_days_row, expiry_target,
                             option_data, option_data1, option_data2):
    """
    Given a date and a position's expiry_target, return which chain to use.
    Matches against mapped monthly/next/3rd monthly for that date.
    If mismatch (rollover), we still map to the correct file by comparing dates.
    """
    me = pd.to_datetime(mapped_days_row['MonthlyExpiry'])
    ne = pd.to_datetime(mapped_days_row['Next Monthly Expiry'])
    fe = pd.to_datetime(mapped_days_row['3rd Monthly Expiry'])

    if pd.to_datetime(expiry_target).date() == me.date():
        return option_data, "monthly"
    if pd.to_datetime(expiry_target).date() == ne.date():
        return option_data1, "next"
    if pd.to_datetime(expiry_target).date() == fe.date():
        return option_data2, "far"

    # If none match, choose the closest future expiry file (or fallback to monthly)
    # This handles mid-month rolls where tiers shift.
    candidates = [(me, option_data, "monthly"), (ne, option_data1, "next"), (fe, option_data2, "far")]
    # pick by absolute date difference
    chosen = min(candidates, key=lambda t: abs((pd.to_datetime(expiry_target).date() - t[0].date()).days))
    return chosen[1], chosen[2]


# ============== Delta-driven leg picking =================
def pick_leg_by_delta_with_extension(
    chain_minute_df,
    spot,
    ref_dt,
    target_delta: float,
    optype: str,          # 'put' or 'call'
    direction: str,       # '>=' or '<=' in abs(delta)-space
    base_lower: int,
    base_upper: int,
    max_extra: int = 10,
    bias: str = 'below'   # 'below' (extend strikes farther OTM for puts) or 'above' (for calls)
):
    """
    Try to pick a leg by delta in a base window around ATM; if no match,
    progressively extend the window by 1, up to `max_extra`. The 'bias'
    controls which side extends first when asymmetric.
    """
    assert optype in ('put','call')
    assert direction in ('>=','<=')

    strikes_all = sorted(chain_minute_df['StrikePrice'].dropna().unique().tolist())
    if not strikes_all:
        return None

    # Which side to preferentially extend?
    # - puts usually need more room below ATM
    # - calls usually need more room above ATM
    prefer_below_first = (bias == 'below')

    for extra in range(0, max_extra + 1):
        lower = base_lower + (extra if prefer_below_first else 0)
        upper = base_upper + (0 if prefer_below_first else extra)

        # If we’ve already tried bias-first, try the opposite on the next iteration
        if extra > 0 and prefer_below_first:
            # attempt the opposite extension on alternating steps
            lower2 = base_lower
            upper2 = base_upper + extra
            candidates = []
            for lo, up in [(lower, upper), (lower2, upper2)]:
                window_df = extract_option_chain_slice(
                    chain_minute_df.assign(StrikePrice=pd.to_numeric(chain_minute_df['StrikePrice'], errors='coerce')),
                    spot,
                    optype
                )
                # Rebuild the slice using the custom lower/upper for this attempt
                strikes = sorted(chain_minute_df['StrikePrice'].dropna().unique().tolist())
                from_atm = set(get_strike_window(strikes, spot, lo, up))
                window_df = chain_minute_df[chain_minute_df['StrikePrice'].isin(from_atm)]
                sel = pick_leg_by_delta(window_df, spot, RF_RATE, ref_dt, target_delta, optype, direction)
                if sel is not None:
                    candidates.append(sel)
            if candidates:
                # Pick the best of the two attempts
                # (closest |Δ| to target, then nearest to spot)
                dd = []
                for s in candidates:
                    d = abs(abs(float(s['Delta'])) - abs(target_delta))
                    dist = abs(float(s['StrikePrice']) - spot)
                    dd.append((d, dist, s))
                dd.sort(key=lambda x: (x[0], x[1]))
                return dd[0][2]
        else:
            # Single attempt (extend one side)
            strikes = sorted(chain_minute_df['StrikePrice'].dropna().unique().tolist())
            from_atm = set(get_strike_window(strikes, spot, lower, upper))
            window_df = chain_minute_df[chain_minute_df['StrikePrice'].isin(from_atm)]
            sel = pick_leg_by_delta(window_df, spot, RF_RATE, ref_dt, target_delta, optype, direction)
            if sel is not None:
                return sel

    return None
# ============== Minute-level snapshotting =================
def _roles_for_type(stype):
    return ('long1','short','long2') if str(stype).lower() == 'rhino' else ('cal_short','cal_long')

def _role_kind_map(stype):
    # leg kind per role
    return {'long1':'put','short':'put','long2':'put'} if str(stype).lower()=='rhino' else {'cal_short':'call','cal_long':'call'}

def _role_lots_map(stype):
    return {'long1': RHINO_LONG_LOTS, 'short': RHINO_SHORT_LOTS, 'long2': RHINO_LONG_LOTS} if str(stype).lower()=='rhino' \
           else {'cal_short': CALENDAR_LOTS, 'cal_long': CALENDAR_LOTS}

def _last_known_legset_for_row(row):
    stype = row['Type']
    roles = _roles_for_type(stype)
    legs = {}
    for r in roles:
        kS = f"{r}_Strike"
        kE = f"{r}_Expiry"  # <-- new
        if kS in row and pd.notna(row[kS]):
            legs[r] = {'strike': float(row[kS])}
            if kE in row and pd.notna(row[kE]):
                legs[r]['expiry'] = pd.to_datetime(row[kE])  # <-- persist exact expiry
    return legs

def _infer_expiry_for_pid(pid: str):
    # PID-YYYYMMDD -> Timestamp
    try:
        dt = pd.to_datetime(pid.replace('PID-',''), format='%Y%m%d')
        return dt
    except Exception:
        return None

def _snapshot_legs_at(dt_date, tstr, pid, stype, legs, mapped_days_row,option_data, option_data1, option_data2, spot, strict_minute=False):
    pid_expiry = _infer_expiry_for_pid(pid)
    if pid_expiry is None:
        return {}

    me = pd.to_datetime(mapped_days_row['MonthlyExpiry'])
    ne = pd.to_datetime(mapped_days_row['Next Monthly Expiry'])
    fe = pd.to_datetime(mapped_days_row['3rd Monthly Expiry'])

    def expiry_for_role(role):
        if stype == 'rhino':
            return pid_expiry
        if role == 'cal_short':
            return pid_expiry
        return (ne if pid_expiry.date() == me.date() else fe)

    kinds = _role_kind_map(stype)
    snap = {}

    for role, meta in legs.items():
        strike = float(meta['strike'])
        exp_target = meta.get('expiry') or expiry_for_role(role)

        # cached tier + O(1) minute slice
        df, tier = resolve_chain_for_expiry_cached(
            dt_date, mapped_days_row, exp_target,
            option_data, option_data1, option_data2
        )

        ch_min = get_chain_minute_fast(tier, dt_date, tstr,fallback=(not strict_minute))
        if ch_min is None or ch_min.empty:
            continue

        kind = kinds[role]
        srow = snapshot_row(ch_min, pd.to_datetime(f"{dt_date} {tstr}"), exp_target, strike, kind, spot, RF_RATE)
        if srow:
            snap[role] = srow

    return snap

def _apply_snapshot_to_row_inplace(df: pd.DataFrame, idx, snap: dict):
    """
    Write snapshot values into row `idx`:
    - Numeric fields: strike, delta, price
    - Datetime field: Expiry (capital E in the snapshot)
    Ensures *_Expiry columns are datetime64[ns] before assignment.
    """
    field_map = [
        ("Strike", "strike"),
        ("Delta",  "delta"),
        ("Price",  "price"),
        ("Expiry", "Expiry"),  # NOTE: capital E matches snapshot key
    ]

    for role, vals in snap.items():
        if not vals:
            continue

        for df_suffix, snap_key in field_map:
            col = f"{role}_{df_suffix}"
            if col not in df.columns:
                continue

            if df_suffix == "Expiry":
                # Ensure the destination column is datetime dtype
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                v = pd.to_datetime(vals.get("Expiry", pd.NaT), errors="coerce")
            else:
                v = vals.get(snap_key, np.nan)

            # Only write meaningful values
            if v is None:
                continue
            try:
                if pd.isna(v):
                    continue
            except Exception:
                pass

            df.at[idx, col] = v
# ===================== Main engine =====================
def generate_rhino_strategy(mapped_days, option_data, option_data1, option_data2, index_df, vix_df):
    # Normalize time strings
    option_data = normalize_time_column(option_data)
    option_data1 = normalize_time_column(option_data1)
    option_data2 = normalize_time_column(option_data2)
    index_df = normalize_time_column(index_df)

    def get_minute_for_leg(lg, date, time_str):
        # lg is a Leg(...) with lg.expiry and lg.kind already set
        chain_df, _ = resolve_chain_for_expiry(
            date, mdrow, lg.expiry, option_data, option_data1, option_data2
        )
        return get_chain_minute(chain_df, date, time_str)

    # Ensure Date types
    for df in [option_data, option_data1, option_data2, index_df, vix_df, mapped_days]:
        df['Date'] = pd.to_datetime(df['Date'])

    trades = []
    open_structs = []  # list[Structure]
    _logged_day_open, _logged_day_eod = set(), set()
    pending_reopen = {}   # { pid: {'target_expiry': Timestamp} }

    # NEW — running realised PnL per PID and the 5-minute stream rows
    realised_by_pid = defaultdict(float)   # PID -> cumulative realised PnL
    pnl_stream_rows = []                   # rows for the CSV you want

    last_tier_by_sid = {}  # place near `open_structs = []`
    chain_minute_cache = {}
    
    
    def log_day_marker(action_dt, tag, vix_reg=None, mdrow=None):
        """
        tag: "DAY_OPEN" or "DAY_EOD"
        vix_reg: "LOW"/"HIGH" if known, else None
        mdrow: mapped_days row (to show ME/NE/FE in comment)
        """
        me = pd.to_datetime(mdrow['MonthlyExpiry']).date()        if mdrow is not None else None
        ne = pd.to_datetime(mdrow['Next Monthly Expiry']).date()  if mdrow is not None else None
        fe = pd.to_datetime(mdrow['3rd Monthly Expiry']).date()   if mdrow is not None else None
        comment = f"ME={me}, NE={ne}, FE={fe}"
        trades.append({
            'Date': pd.to_datetime(action_dt).date(),
            'Time': pd.to_datetime(action_dt).strftime("%H:%M"),
            'StructureID': "DAY",      # marker rows
            'Slot': "",
            'Type': "marker",
            'Action': tag,             # "DAY_OPEN" or "DAY_EOD"
            'LegRole': None,
            'Strike': None,
            'Delta': None,
            'VIX_Regime': vix_reg,
            'DTE': None,
            'Comment': comment
        })

    # Helper to log actions
    def log(action_dt, pid, slot, stype, action,
        leg_role=None, strike=None, delta=None, vix_reg=None, dte=None,
        comment="", legs_snapshot=None, spot=None,pnl=None,reason=None):   # <— add spot
        row = {
            'Date': pd.to_datetime(action_dt).date(),
            'StructureID': pid,
            'Time': pd.to_datetime(action_dt).strftime("%H:%M"),
            'Spot': spot,
            'Slot': slot,
            'Type': stype,
            'Action': action,
            'LegRole': leg_role,
            'Strike': strike,
            'Delta': delta,
            'VIX_Regime': vix_reg,
            'DTE': dte,
            'Comment': comment
        }
        if spot is not None:
            row['Spot'] = float(spot)
        if pnl is not None:
            row['PnL'] = float(pnl)
        if reason is not None:
            row['Reason'] = reason

        if legs_snapshot:
            for key, vals in legs_snapshot.items():
                if vals is None:
                    continue
                row[f"{key}_Strike"] = vals.get("strike")
                row[f"{key}_Delta"]  = vals.get("delta")
                if 'price' in vals:
                    row[f"{key}_Price"]  = vals.get("price")
                if 'Expiry' in vals:                                   # <-- add this
                    row[f"{key}_Expiry"] = vals.get("expiry") 
        trades.append(row)
    def exlog(ts, pid, slot, reason, extra=""):
        trades.append({
            'Date': pd.to_datetime(ts).date(),
            'Time': pd.to_datetime(ts).strftime("%H:%M"),
            'StructureID': pid,
            'Slot': slot,
            'Type': "exception",
            'Action': reason,
            'Comment': extra
        })
    # Iterate index minutes
    # prefilter to 5-min rows once
    idx5 = index_df[index_df['Time'].str.match(r'^\d{2}:\d{2}$', na=False)]
    idx5 = idx5[idx5['Time'].str[-2:].isin(['00','05','10','15','20','25','30','35','40','45','50','55'])]
    
    def get_chain_minute(df, dt_date, tstr):
        key = (id(df), dt_date, tstr)  # df identity + date + time
        if key in chain_minute_cache:
            return chain_minute_cache[key]
        out = df[(df['Date'].dt.date == dt_date) & (df['Time'] == tstr)].copy()
        chain_minute_cache[key] = out
        return out

    for irow in idx5.itertuples(index=False):
        date = pd.to_datetime(irow.Date).date()
        time_str = irow.Time
        spot = float(irow.Spot)

        ref_dt = pd.to_datetime(f"{date} {time_str}")
       


        # Map row for this date
        mdrow = mapped_days[mapped_days['Date'].dt.date == date]
        if mdrow.empty:
            continue
        mdrow = mdrow.iloc[0]

        # VIX regime for the day
        # VIX regime for the day (SAFE)
        # VIX regime for the day  (SAFE VERSION)
        vrow = vix_df[vix_df['Date'].dt.date == date]
        if vrow.empty:
            # no VIX row for this date — skip this minute to avoid undefined vix values
            continue

        try:
            vix_pct_val = float(vrow.iloc[0]['VIX_Percentile_100D'])
        except Exception:
            vix_pct_val = float('nan')

        round_vix = None if pd.isna(vix_pct_val) else round(vix_pct_val, 1)
        low_vol = (not pd.isna(vix_pct_val)) and (vix_pct_val < 75.0)
        regime_tag = "LOW" if low_vol else "HIGH"

        # --- Day markers (logged once per date) ---
        if (time_str == DAY_OPEN_TIME) and (date not in _logged_day_open):
            log_day_marker(pd.to_datetime(f"{date} {time_str}"), "DAY_OPEN", regime_tag, mdrow=mdrow)
            _logged_day_open.add(date)
            # 👇 add this
            print(f"▶️ Processing {date} ...")

        if (time_str == DAY_EOD_TIME) and (date not in _logged_day_eod):
            log_day_marker(pd.to_datetime(f"{date} {time_str}"), "DAY_EOD", regime_tag, mdrow=mdrow)
            _logged_day_eod.add(date)
            # 👇 add this
            print(f"✅ {date} done.")

        
        # - ------- Universal Exit with 15:00 timing --------
        for st in list(open_structs):
            chain_for_st, tier = resolve_chain_for_expiry(
                date, mdrow, st.expiry, option_data, option_data1, option_data2
            )

            # --- tier-shift debug (st exists here) ---
            prev = last_tier_by_sid.get(st.sid)
            if prev != tier:
                dbg("TIER_SHIFT", sid=st.sid, pid=st.pid, old=prev, new=tier, date=str(date), time=time_str)
                last_tier_by_sid[st.sid] = tier
            # -----------------------------------------

            _T, DTE = calculate_time_to_expiry(st.expiry, ref_dt)

            # On the day it becomes 15, exit at 15:00 or later.
            def ue_due(dte, tstr):
                if dte < UNIVERSAL_EXIT_DTE:
                    return True
                if dte == UNIVERSAL_EXIT_DTE and tstr >= "15:00":
                    return True
                return False

            if not ue_due(DTE, time_str):
                continue
            
            # Try to snapshot current leg strikes + live deltas (using current minute price) — CACHED
            snap = {}
            roles = _roles_for_type(st.stype)
            for role in roles:
                lg = next((x for x in st.legs if x.role == role), None)
                if lg is None:
                    continue
                # cached tier + O(1) minute rows
                _, tier_ue = resolve_chain_for_expiry_cached(date, mdrow, lg.expiry, option_data, option_data1, option_data2)
                ch_min_leg = get_chain_minute_fast(tier_ue, date, time_str)
                if ch_min_leg.empty:
                    continue
                srow = snapshot_row(ch_min_leg, pd.to_datetime(f"{date} {time_str}"), lg.expiry, lg.strike, lg.kind, spot, RF_RATE)
                if srow:
                    snap[role] = srow

            # compute PnL *after* snap is ready
            pnl_value = structure_pnl(st, snap)
            realised_by_pid[st.pid] += float(pnl_value)       # NEW


            st.status = "closed"

            # free the slot
            ensure_group(st.pid)
            if st.stype == "rhino" and st.slot in ("R1","R2"):
                position_groups[st.pid][st.slot] = None
            if st.stype == "calendar" and st.slot == "CAL":
                position_groups[st.pid]["CAL"] = None

            open_structs.remove(st)

            # single log row with PnL
            log(ref_dt, st.pid, st.slot, st.stype, "UNIVERSAL_EXIT(<15DTE)",
                vix_reg=regime_tag, dte=DTE, comment=f"tier={tier}",
                legs_snapshot=snap, spot=spot, pnl=pnl_value)

        # Entry: first Monday 09:30 (only if we still can add)
        if is_first_monday_930(date, time_str):
            # Rhino must be entered on NEXT-MONTH expiry (option_data1)
            next_expiry = pd.to_datetime(mdrow['Next Monthly Expiry'])
            pid = make_pid(next_expiry)
            ensure_group(pid)

            if rhino_slots_count(pid) < 2:
                slot = first_free_rhino_slot(pid)  # "R1" or "R2"
                dbg("ENTRY_CHECK",
                    date=str(date), time=time_str, pid=pid, slot=slot,
                    chain="option_data1 (next)", next_expiry=str(next_expiry.date()))
                if slot is not None:
                    chain = get_chain_minute(option_data1, date, time_str)
                    if chain.empty:
                        exlog(ref_dt, pid, slot, "NO_CHAIN_MINUTE",
                            "entry: option_data1 (next); time="+time_str)
                    else:
                        slice_df = extract_option_chain_slice(chain, spot, 'put')
                        
                        # 👇 add this just before selecting legs
                        _debug_print_candidates(chain, spot, ref_dt)


                        if slice_df.empty:
                            exlog(ref_dt, pid, slot, "NO_STRIKE_WINDOW", f"spot={spot}")
                        else:
                            sel1 = pick_leg_by_delta_with_extension(
                                chain, spot, ref_dt,
                                DELTA_TARGETS_RHINO['leg1_long'], 'put', '>=',
                                base_lower=PUT_WIN_BELOW, base_upper=PUT_WIN_ABOVE,
                                max_extra=10, bias='below'
                            )
                            sel2 = pick_leg_by_delta_with_extension(
                                chain, spot, ref_dt,
                                DELTA_TARGETS_RHINO['leg2_short'], 'put', '>=',
                                base_lower=PUT_WIN_BELOW, base_upper=PUT_WIN_ABOVE,
                                max_extra=10, bias='below'
                            )
                            sel3 = pick_leg_by_delta_with_extension(
                                chain, spot, ref_dt,
                                DELTA_TARGETS_RHINO['leg3_long'], 'put', '>=',
                                base_lower=PUT_WIN_BELOW, base_upper=PUT_WIN_ABOVE,
                                max_extra=10, bias='below'
                            )

                            if any(s is None for s in (sel1, sel2, sel3)):
                                # optional: log which one failed so you can see why no entry happened
                                exlog(ref_dt, pid, slot, "DELTA_PICK_FAILED",
                                    f"sel1={sel1 is None}, sel2={sel2 is None}, sel3={sel3 is None}")
                            else:
                                sid = f"{pid}-{slot}"
                                rh = Structure(
                                    sid=sid, stype="rhino", opened_at=ref_dt, expiry=next_expiry,
                                    legs=[
                                        Leg("put","long1", float(sel1['StrikePrice']), pd.to_datetime(sel1['ExpiryDate']),
                                            RHINO_LONG_LOTS, float(sel1['Open']), +1),
                                        Leg("put","short", float(sel2['StrikePrice']), pd.to_datetime(sel2['ExpiryDate']),
                                            RHINO_SHORT_LOTS, float(sel2['Open']), -1),
                                        Leg("put","long2", float(sel3['StrikePrice']), pd.to_datetime(sel3['ExpiryDate']),
                                            RHINO_LONG_LOTS, float(sel3['Open']), +1),
                                    ],
                                    pid=pid, slot=slot
                                )

                                open_structs.append(rh)
                                position_groups[pid][slot] = rh.sid
                                legs_snap = {
                                    "long1": {"strike": float(sel1['StrikePrice']),
                                            "delta":  float(sel1['Delta']),
                                            "price":  float(sel1['Open']),
                                            "expiry": pd.to_datetime(sel1['ExpiryDate']).date()},                                            
                                    "short": {"strike": float(sel2['StrikePrice']),
                                            "delta":  float(sel2['Delta']),
                                            "price":  float(sel2['Open']),
                                            "expiry": pd.to_datetime(sel2['ExpiryDate']).date()},                                            
                                    "long2": {"strike": float(sel3['StrikePrice']),
                                            "delta":  float(sel3['Delta']),
                                            "price":  float(sel3['Open']),
                                            "expiry": pd.to_datetime(sel3['ExpiryDate']).date()},                                            
                                }

                                log(ref_dt, pid, slot, rh.stype, "OPEN",
                                    vix_reg=regime_tag,
                                    comment=f"exp={next_expiry.date()}, VIX={round_vix}",
                                    legs_snapshot=legs_snap, spot=spot)

                # else: no rows at that minute; skip

        # ============= Monitoring & Adjustments =============
        # For each open structure, align chain daily and then check conditions.
        minute_unreal_by_pid = defaultdict(float)

        _reopen_done = False  # ensure we process pending_reopen at most once per minute

        for st in list(open_structs):
            if st.status != "open":
                continue

            chain_for_st, tier = resolve_chain_for_expiry(date, mdrow, st.expiry, option_data, option_data1, option_data2)

            # --- tier-shift debug (st exists here) ---
            prev = last_tier_by_sid.get(st.sid)
            if prev != tier:
                dbg("TIER_SHIFT", sid=st.sid, pid=st.pid, old=prev, new=tier, date=str(date), time=time_str)
                last_tier_by_sid[st.sid] = tier
            # -----------------------------------------

            # Minute rows for this st
            leg_info = {}
            for lg in st.legs:
                # resolve the correct tier once for this leg’s expiry
                _, tier_leg = resolve_chain_for_expiry_cached(
                    date, mdrow, lg.expiry, option_data, option_data1, option_data2
                )
                ch_min_leg = get_chain_minute_fast(tier_leg, date, time_str, fallback=True)
                if ch_min_leg.empty: 
                    continue
                row = ch_min_leg[(ch_min_leg['StrikePrice'] == lg.strike) &
                                (pd.to_datetime(ch_min_leg['ExpiryDate']).dt.date == lg.expiry.date()) &
                                (ch_min_leg['Type'].str.lower() == lg.kind)]
                if row.empty:
                    continue
                r0 = row.iloc[0]
                T, DTE = calculate_time_to_expiry(lg.expiry, ref_dt)
                price = float(r0["Open"])
                iv = iv_cached(price, spot, lg.strike, T, RF_RATE, lg.kind)
                d  = delta_cached(spot, lg.strike, T, RF_RATE, iv, lg.kind)
                # inside: for lg in st.legs: ... after you compute price, iv, d
                # AFTER (lowercase keys to match snapshot_row + structure_pnl)
                leg_info[lg.role] = {
                    'price':       price,
                    'delta':       d,
                    'dte':         DTE,
                    'entry_price': float(lg.entry_price),
                    'lots':        int(lg.lots),
                    'dir':         int(lg.direction),
                    'strike':      float(lg.strike),
                    'expiry':      lg.expiry.date(),
                    'kind':        lg.kind
                }


            # If not all legs found, skip this minute
            if st.stype == "rhino" and not all(k in leg_info for k in ["long1","short","long2"]):
                continue
            if st.stype == "calendar" and not all(k in leg_info for k in ["cal_short","cal_long"]):
                continue
            
            # Only contributes if the structure is still OPEN at this minute.
            if st.status == "open":
                mtm_s = 0.0
                for lg in st.legs:
                    role = lg.role
                    px_now = float(leg_info[role]['price'])  # computed above for this minute
                    mtm_s += (px_now - float(lg.entry_price)) * float(lg.direction) * int(lg.lots) * LOT_SIZE
                minute_unreal_by_pid[st.pid] += float(mtm_s)


            # Universal DTE check already handled above; now regime rules:

            # ---- Calendar independent exit ----
            if st.stype == "calendar":
                sd = leg_info["cal_short"]["delta"]
                cond_up = sd is not None and sd >= 0.50
                cond_dn = sd is not None and sd <= 0.30
                key = "CAL_EXIT"

                if cond_up or cond_dn:
                    st.bump_counter(key)
                    if st.reached_persist(key):

                        # snapshot both legs with prices/delta at this minute — CACHED
                        snap = {}
                        for role in ("cal_short", "cal_long"):
                            lg = next((x for x in st.legs if x.role == role), None)
                            if lg is None:
                                continue
                            _, tier_cal = resolve_chain_for_expiry_cached(date, mdrow, lg.expiry, option_data, option_data1, option_data2)
                            ch_min_leg = get_chain_minute_fast(tier_cal, date, time_str)
                            if ch_min_leg.empty:
                                continue
                            srow = snapshot_row(ch_min_leg, pd.to_datetime(f"{date} {time_str}"), lg.expiry, lg.strike, lg.kind, spot, RF_RATE)
                            if srow:
                                snap[role] = srow


                        # compute P&L after we have both legs
                        pnl_value = structure_pnl(st, snap)
                        realised_by_pid[st.pid] += float(pnl_value)       # NEW


                        # mark closed + free CAL slot
                        st.status = "closed"
                        ensure_group(st.pid)
                        position_groups[st.pid]["CAL"] = None
                        open_structs.remove(st)

                        # log one EXIT row with details
                        cal_short_strike = next((x.strike for x in st.legs if x.role == "cal_short"), None)
                        reason_txt = "cal_short Δ>=0.50" if cond_up else "cal_short Δ<=0.30"
                        log(
                            ref_dt, st.pid, st.slot, st.stype,
                            "EXIT(cal_short delta outside [0.30,0.50])", "cal_short",
                            strike=cal_short_strike,
                            delta=float(sd) if sd is not None else None,
                            vix_reg=regime_tag,
                            dte=leg_info["cal_short"]["dte"],
                            comment=f"tier={tier}; reason={reason_txt}",
                            legs_snapshot=snap,
                            spot=spot,
                            pnl=pnl_value,
                            reason="CAL_EXIT"
                        )
                        continue
                else:
                    st.reset_counter(key)

                # skip other rhino logic for calendars
                continue

                

            # ---- Rhino regime-based conditions ----
            # aliases
            short_delta = leg_info["short"]["delta"]
            long_delta  = leg_info["long1"]["delta"]

            # Helper to add another Rhino (if <2)
            def try_add_rhino(target_expiry=None,trigger_reason=None):
                nonlocal date, time_str, mdrow, spot, ref_dt, regime_tag, round_vix
                if target_expiry is None:
                    target_expiry = pd.to_datetime(mdrow['Next Monthly Expiry'])

                pid = make_pid(target_expiry)
                ensure_group(pid)
                slot = first_free_rhino_slot(pid)
                if slot is None:
                    exlog(ref_dt, pid, None, "NO_FREE_SLOT", "pid already has 2 rhinos")
                    return False

                chain_all, tier = resolve_chain_for_expiry(
                    date, mdrow, target_expiry, option_data, option_data1, option_data2
                )

                # --- robust minute pick on the same date ---
                day_slice = chain_all[chain_all['Date'].dt.date == date]
                if day_slice.empty:
                    exlog(ref_dt, pid, slot, "NO_DAY_DATA", f"tier={tier}")
                    return False

                # exact minute first
                chain = day_slice[day_slice['Time'] == time_str].copy()
                if chain.empty:
                    # find nearest earlier time on the day
                    le = day_slice[day_slice['Time'] <= time_str].sort_values('Time')
                    if not le.empty:
                        fallback_time = le.iloc[-1]['Time']
                    else:
                        # otherwise earliest time on day
                        ge = day_slice.sort_values('Time')
                        if ge.empty:
                            exlog(ref_dt, pid, slot, "NO_CHAIN_MINUTE", f"tier={tier} time={time_str}")
                            return False
                        fallback_time = ge.iloc[0]['Time']

                    # <-- crucial: pick the FULL chain for that fallback minute
                    chain = day_slice[day_slice['Time'] == fallback_time].copy()
                _debug_print_candidates(chain, spot, ref_dt)
                # --- after you compute `day_slice` and `chain`, insert this: ---
                label = f"REENTER-RHINO tier={tier}"
                try:
                    _debug_print_candidate_counts(
                        chain_minute_df=chain,
                        chain_all_df=chain_all,
                        spot=spot,
                        ref_dt=ref_dt,
                        label=label,
                        exp_target=target_expiry,   
                        show_missing=True
                    )
                except Exception as e:
                    print(f"[CANDS-COUNT] error: {e}")
                # --------------------------------------------------------------

                # -------------------------------------------

                slice_df = extract_option_chain_slice(chain, spot, 'put')

                sel1 = pick_leg_by_delta_with_extension(
                    chain, spot, ref_dt,
                    DELTA_TARGETS_RHINO['leg1_long'], 'put', '>=',
                    PUT_WIN_BELOW, PUT_WIN_ABOVE, 10, 'below'
                )
                sel2 = pick_leg_by_delta_with_extension(
                    chain, spot, ref_dt,
                    DELTA_TARGETS_RHINO['leg2_short'], 'put', '>=',
                    PUT_WIN_BELOW, PUT_WIN_ABOVE, 10, 'below'
                )
                sel3 = pick_leg_by_delta_with_extension(
                    chain, spot, ref_dt,
                    DELTA_TARGETS_RHINO['leg3_long'], 'put', '>=',
                    PUT_WIN_BELOW, PUT_WIN_ABOVE, 10, 'below'
                )

                sel1 = sel1.to_dict() if sel1 is not None else None
                sel2 = sel2.to_dict() if sel2 is not None else None
                sel3 = sel3.to_dict() if sel3 is not None else None

                if sel1 is None:
                    exlog(ref_dt, pid, slot, "NO_DELTA_MATCH_long1", "|Δ|>=0.50 not found")
                    return False
                if sel2 is None:
                    exlog(ref_dt, pid, slot, "NO_DELTA_MATCH_short", "|Δ|>=0.35 not found")
                    return False
                if sel3 is None:
                    exlog(ref_dt, pid, slot, "NO_DELTA_MATCH_long2", "|Δ|>=0.18 not found")
                    return False

                sid = f"{pid}-{slot}"
                rh = Structure(
                    sid=sid, stype="rhino", opened_at=ref_dt, expiry=pd.to_datetime(target_expiry),
                    legs=[
                        Leg("put","long1", float(sel1['StrikePrice']), pd.to_datetime(sel1['ExpiryDate']), RHINO_LONG_LOTS, float(sel1['Open']), +1),
                        Leg("put","short", float(sel2['StrikePrice']), pd.to_datetime(sel2['ExpiryDate']), RHINO_SHORT_LOTS, float(sel2['Open']), -1),
                        Leg("put","long2", float(sel3['StrikePrice']), pd.to_datetime(sel3['ExpiryDate']), RHINO_LONG_LOTS, float(sel3['Open']), +1),
                    ],
                    pid=pid, slot=slot
                )
                open_structs.append(rh)
                position_groups[pid][slot] = rh.sid

                legs_snap = {
                    "long1": {"strike": float(sel1['StrikePrice']),
                            "delta":  float(sel1['Delta']),
                            "price":  float(sel1['Open']),
                            "expiry": pd.to_datetime(sel1['ExpiryDate']).date()
                            },
                    "short": {"strike": float(sel2['StrikePrice']),
                            "delta":  float(sel2['Delta']),
                            "price":  float(sel2['Open']),
                            "expiry": pd.to_datetime(sel2['ExpiryDate']).date()
                            },
                    "long2": {"strike": float(sel3['StrikePrice']),
                            "delta":  float(sel3['Delta']),
                            "price":  float(sel3['Open']),
                            "expiry": pd.to_datetime(sel3['ExpiryDate']).date()
                            },
                }
                log(ref_dt, pid, slot, rh.stype, "OPEN", vix_reg=regime_tag,
                    comment=f"tier={tier}, exp={pd.to_datetime(target_expiry).date()}, VIX={round_vix}",
                    legs_snapshot=legs_snap)

                return True


            def try_add_calendar(trigger_reason=None):
                pid = make_pid(st.expiry)  # calendar belongs to SAME expiry as the rhino it hedges
                ensure_group(pid)
                if calendar_exists(pid):
                    exlog(ref_dt, pid, "CAL", "CAL_EXISTS", "calendar already open")

                    return False
                

                same_exp = st.expiry
                next_exp = (pd.to_datetime(mdrow['Next Monthly Expiry'])
                            if same_exp.date() == pd.to_datetime(mdrow['MonthlyExpiry']).date()
                            else pd.to_datetime(mdrow['3rd Monthly Expiry']))

                ch_same = get_chain_minute(chain_for_st, date, time_str)
                ch_next_df, _tier2 = resolve_chain_for_expiry(date, mdrow, next_exp, option_data, option_data1, option_data2)
                ch_next = get_chain_minute(ch_next_df, date, time_str)
                
                if ch_same.empty:
                    exlog(ref_dt, pid, "CAL", "NO_CHAIN_CAL_SAME", f"time={time_str}")
                    return False
                if ch_next.empty:
                    exlog(ref_dt, pid, "CAL", "NO_CHAIN_CAL_NEXT", f"time={time_str}")
                    return False
                
                sl1 = pick_leg_by_delta_with_extension(
                    ch_same, spot, ref_dt,
                    DELTA_TARGET_CAL, 'call', '<=',
                    base_lower=CALL_WIN_BELOW, base_upper=CALL_WIN_ABOVE,
                    max_extra=10, bias='above'   # calls: extend upwards first
                )
                sl2 = pick_leg_by_delta_with_extension(
                    ch_next, spot, ref_dt,
                    DELTA_TARGET_CAL, 'call', '<=',
                    base_lower=CALL_WIN_BELOW, base_upper=CALL_WIN_ABOVE,
                    max_extra=10, bias='above'
                )


                sl1 = sl1.to_dict() if sl1 is not None else None
                sl2 = sl2.to_dict() if sl2 is not None else None

                if sl1 is None:
                    exlog(ref_dt, pid, "CAL", "NO_DELTA_MATCH_CAL_short", "|Δ|<=0.40 not found (same exp)")
                    return False
                if sl2 is None:
                    exlog(ref_dt, pid, "CAL", "NO_DELTA_MATCH_CAL_long", "|Δ|<=0.40 not found (next exp)")
                    return False

                sid = f"{pid}-CAL"
                cal = Structure(
                    sid=sid, stype="calendar", opened_at=ref_dt, expiry=same_exp,
                    legs=[
                        Leg("call","cal_short", float(sl1['StrikePrice']), pd.to_datetime(sl1['ExpiryDate']),
                            CALENDAR_LOTS, float(sl1['Open']), -1),
                        Leg("call","cal_long", float(sl2['StrikePrice']), pd.to_datetime(sl2['ExpiryDate']),
                            CALENDAR_LOTS, float(sl2['Open']), +1),
                    ],
                    pid=pid, slot="CAL"
                )
                open_structs.append(cal)
                position_groups[pid]["CAL"] = cal.sid

                legs_snap = {
                    "cal_short": {"strike": float(sl1['StrikePrice']),
                                "delta":  float(sl1['Delta']),
                                "price":  float(sl1['Open']),
                                "expiry": pd.to_datetime(sl1['ExpiryDate']).date()
                                },
                    "cal_long":  {"strike": float(sl2['StrikePrice']),
                                "delta":  float(sl2['Delta']),
                                "price":  float(sl2['Open']),
                                "expiry": pd.to_datetime(sl2['ExpiryDate']).date()
                                },
                }
                log(ref_dt, pid, "CAL", cal.stype, "OPEN", vix_reg=regime_tag,
                    comment=f"Calendar added; same_exp={same_exp.date()}, next_exp={next_exp.date()}, VIX={round_vix}",
                    legs_snapshot=legs_snap,reason=trigger_reason)
                return True

            # ---- reattempt any pending re-opens (do this once per minute, after try_add_rhino exists) ----
            if (not _reopen_done) and pending_reopen:
                for pid, info in list(pending_reopen.items()):
                    tgt = pd.to_datetime(info['target_expiry'])
                    _T, _dte = calculate_time_to_expiry(tgt, ref_dt)
                    # if already at/below 15 DTE and it's 15:00 or later, give up on re-open
                    if _dte < UNIVERSAL_EXIT_DTE or (_dte == UNIVERSAL_EXIT_DTE and time_str >= "15:00"):
                        pending_reopen.pop(pid, None)
                        continue
                    ok = try_add_rhino(target_expiry=tgt)
                    if ok:
                        pending_reopen.pop(pid, None)
                _reopen_done = True
            # ----------------------------------------------------------------------------------------------

            # ---- Apply regime rules with persistence ----
            # ---- Rhino regime-based conditions ----
            short_delta = leg_info["short"]["delta"]
            long_delta  = leg_info["long1"]["delta"]

            # condition keys
            k_sd_45_add   = "SD<=-0.45_ADD"
            k_sd_52_exit  = "SD<=-0.52_EXIT"
            k_ld_45_add   = "LD>=-0.45_ADD"
            k_ld_40_reset = "LD>=-0.40_RESET"   # universal exit

            # Evaluate once
            cond_sd_45 = (short_delta is not None) and (short_delta <= -0.45)
            cond_sd_52 = (short_delta is not None) and (short_delta <= -0.52)
            cond_ld_45 = (long_delta  is not None)  and (long_delta  >= -0.45)
            cond_ld_40 = (long_delta  is not None)  and (long_delta  >= -0.40)

            if low_vol:
                # --- Short-delta rules (same as before) ---
                if cond_sd_45:
                    st.bump_counter(k_sd_45_add)
                    if st.reached_persist(k_sd_45_add):
                        pid_here = st.pid
                        if rhino_slots_count(st.pid) == 1:
                            try_add_rhino(trigger_reason="Added_rhino_from_SD<=-0.45")
                else:
                    st.reset_counter(k_sd_45_add)

                if cond_sd_52:
                    st.bump_counter(k_sd_52_exit)
                    if st.reached_persist(k_sd_52_exit):
                        # snapshot + exit + reopen same-expiry
                        snap = {}
                        for role in ("long1", "short", "long2"):
                            if role in leg_info:
                                snap[role] = {
                                    "strike": next((x.strike for x in st.legs if x.role == role), None),
                                    "delta": float(leg_info[role]['delta']),
                                    "price": float(leg_info[role]['price'])
                                }
                        pnl_value = structure_pnl(st, snap)
                        realised_by_pid[st.pid] += float(pnl_value)       # NEW

                        st.status = "closed"
                        short_strike = next((x.strike for x in st.legs if x.role == "short"), None)

                        log(ref_dt, st.pid, st.slot, st.stype, "EXIT(delta_short<=-0.52)", "short",
                            strike=short_strike, delta=float(leg_info["short"]["delta"]),
                            vix_reg=regime_tag, dte=leg_info["short"]["dte"],
                            comment=f"tier={tier}", legs_snapshot=snap, spot=spot, pnl=pnl_value)

                        ensure_group(st.pid)
                        if st.slot in ("R1", "R2"):
                            position_groups[st.pid][st.slot] = None
                        open_structs.remove(st)

                        ok = try_add_rhino(target_expiry=st.expiry)
                        if not ok:
                            pending_reopen[make_pid(st.expiry)] = {'target_expiry': st.expiry}
                        # after EXIT done for this st, continue loop
                        continue
                else:
                    st.reset_counter(k_sd_52_exit)

                # --- Long-delta -0.45 first (ADD/Calendar), THEN -0.40 universal exit ---
                if cond_ld_45:
                    st.bump_counter(k_ld_45_add)
                    if st.reached_persist(k_ld_45_add):
                        log(ref_dt, st.pid, st.slot, st.stype,
                            "TRIGGER(LD>=-0.45)",
                            leg_role="long1",
                            delta=float(leg_info["long1"]["delta"]),
                            vix_reg=regime_tag,
                            dte=leg_info["long1"]["dte"],
                            reason="LD>=-0.45_persisted")
                        _Ttmp, DTE_long = calculate_time_to_expiry(st.expiry, ref_dt)
                        if rhino_slots_count(st.pid) < 2:
                            if DTE_long < 25:
                                try_add_calendar(trigger_reason="CAL_from_DTE<25")
                            else:
                                try_add_rhino(trigger_reason='Added_rhino_from_LD>=-0.45')
                else:
                    st.reset_counter(k_ld_45_add)

                if cond_ld_40:
                    st.bump_counter(k_ld_40_reset)
                    if st.reached_persist(k_ld_40_reset):
                        # snapshot + exit + reopen same-expiry
                        snap = {}
                        for role in ("long1","short","long2"):
                            if role in leg_info:
                                snap[role] = {
                                    "strike": next((x.strike for x in st.legs if x.role == role), None),
                                    "delta":  float(leg_info[role]['delta']),
                                    "price": float(leg_info[role]['price'])
                                }
                        st.status = "closed"
                        pnl_value = structure_pnl(st, snap)
                        realised_by_pid[st.pid] += float(pnl_value)       # NEW

                        long1_strike = next((x.strike for x in st.legs if x.role=="long1"), None)

                        log(ref_dt, st.pid, st.slot, st.stype, "EXIT(delta_long>=-0.40)", "long1",
                            strike=long1_strike, delta=float(leg_info["long1"]["delta"]),
                            vix_reg=regime_tag, dte=leg_info["long1"]["dte"],
                            comment=f"tier={tier}", legs_snapshot=snap, spot=spot, pnl=pnl_value)

                        ensure_group(st.pid)
                        if st.slot in ("R1","R2"):
                            position_groups[st.pid][st.slot] = None
                        open_structs.remove(st)

                        ok = try_add_rhino(target_expiry=st.expiry)
                        if not ok:
                            pending_reopen[make_pid(st.expiry)] = {'target_expiry': st.expiry}
                        continue
                else:
                    st.reset_counter(k_ld_40_reset)

            else:
                # -------- HIGH VOL --------
                # SD <= -0.45 → square off and create new rhino (same as before)
                if cond_sd_45:
                    st.bump_counter(k_sd_45_add)
                    if st.reached_persist(k_sd_45_add):
                        snap = {}
                        for role in ("long1", "short", "long2"):
                            if role in leg_info:
                                snap[role] = {
                                    "strike": next((x.strike for x in st.legs if x.role == role), None),
                                    "delta": float(leg_info[role]['delta']),
                                    "price": float(leg_info[role]['price'])
                                }
                        st.status = "closed"
                        pnl_value = structure_pnl(st, snap)
                        realised_by_pid[st.pid] += float(pnl_value)       # NEW

                        short_strike = next((x.strike for x in st.legs if x.role == "short"), None)

                        log(ref_dt, st.pid, st.slot, st.stype, "EXIT(delta_short<=-0.45)", "short",
                            strike=short_strike, delta=float(leg_info["short"]["delta"]),
                            vix_reg=regime_tag, dte=leg_info["short"]["dte"],
                            comment=f"tier={tier}", legs_snapshot=snap, spot=spot, pnl=pnl_value)

                        ensure_group(st.pid)
                        if st.slot in ("R1", "R2"):
                            position_groups[st.pid][st.slot] = None
                        open_structs.remove(st)

                        ok = try_add_rhino(target_expiry=st.expiry)
                        if not ok:
                            pending_reopen[make_pid(st.expiry)] = {'target_expiry': st.expiry}
                        continue
                else:
                    st.reset_counter(k_sd_45_add)

                # --- Long-delta -0.45 first (Calendar when exactly one in PID), THEN -0.40 universal exit ---
                if cond_ld_45:
                    st.bump_counter(k_ld_45_add)
                    if st.reached_persist(k_ld_45_add):
                        pid_here = st.pid
                        log(ref_dt, st.pid, st.slot, st.stype,
                            "TRIGGER(LD>=-0.45)",
                            leg_role="long1",
                            delta=float(leg_info["long1"]["delta"]),
                            vix_reg=regime_tag,
                            dte=leg_info["long1"]["dte"],
                            reason="LD>=-0.45_persisted")
                        _Ttmp, DTE_long = calculate_time_to_expiry(st.expiry, ref_dt)
                        if rhino_slots_count(st.pid) == 1:
                            try_add_calendar(trigger_reason="CAL_from_LD>=-0.45")
                else:
                    st.reset_counter(k_ld_45_add)

                if cond_ld_40:
                    st.bump_counter(k_ld_40_reset)
                    if st.reached_persist(k_ld_40_reset):
                        snap = {}
                        for role in ("long1","short","long2"):
                            if role in leg_info:
                                snap[role] = {
                                    "strike": next((x.strike for x in st.legs if x.role == role), None),
                                    "delta":  float(leg_info[role]['delta']),
                                    "price": float(leg_info[role]['price'])
                                }
                        st.status = "closed"
                        pnl_value = structure_pnl(st, snap)
                        realised_by_pid[st.pid] += float(pnl_value)       # NEW

                        long1_strike = next((x.strike for x in st.legs if x.role=="long1"), None)

                        log(ref_dt, st.pid, st.slot, st.stype, "EXIT(delta_long>=-0.40)", "long1",
                            strike=long1_strike, delta=float(leg_info["long1"]["delta"]),
                            vix_reg=regime_tag, dte=leg_info["long1"]["dte"],
                            comment=f"tier={tier}", legs_snapshot=snap, spot=spot, pnl=pnl_value)

                        ensure_group(st.pid)
                        if st.slot in ("R1","R2"):
                            position_groups[st.pid][st.slot] = None
                        open_structs.remove(st)

                        ok = try_add_rhino(target_expiry=st.expiry)
                        if not ok:
                            pending_reopen[make_pid(st.expiry)] = {'target_expiry': st.expiry}
                        continue
                else:
                    st.reset_counter(k_ld_40_reset)
        
        # NEW — after finishing the for-st loop for this minute, flush per-PID rows
        if 'minute_unreal_by_pid' in locals():
            for pid_unr, unr_val in minute_unreal_by_pid.items():
                real_val = realised_by_pid[pid_unr]  # running realised
                pnl_stream_rows.append({
                    'PID': pid_unr,
                    'Date': date,
                    'Time': time_str,
                    'UnrealisedPnL': float(unr_val),
                    'RealisedPnL': float(real_val),
                    'NetPnL': float(unr_val + real_val),
                })
            # reset for next minute
            del minute_unreal_by_pid


    # --- End of main loop over index minutes ---
    # Final trades DataFrame
    out = pd.DataFrame(trades)
    if not out.empty:
        # the later is for inspection convenience
        # cols = [
        #     'Date','Time','StructureID','Slot','Type','Action','LegRole',
        #     'Strike','Delta','VIX_Regime','DTE','Comment','Spot',          
        #     'long1_Strike','long1_Delta','long1_Price',                    
        #     'short_Strike','short_Delta','short_Price',                    
        #     'long2_Strike','long2_Delta','long2_Price',                    
        #     'cal_short_Strike','cal_short_Delta','cal_short_Price',        
        #     'cal_long_Strike','cal_long_Delta','cal_long_Price'            
        # ]

        num_cols = [
            'long1_Strike','long1_Delta','long1_Price',
            'short_Strike','short_Delta','short_Price',
            'long2_Strike','long2_Delta','long2_Price',
            'cal_short_Strike','cal_short_Delta','cal_short_Price',
            'cal_long_Strike','cal_long_Delta','cal_long_Price',
        ]
        expiry_cols = [
            'long1_Expiry','short_Expiry','long2_Expiry',
            'cal_short_Expiry','cal_long_Expiry',
        ]

        for c in num_cols:
            if c not in out.columns:
                out[c] = np.nan

        for c in expiry_cols:
            if c not in out.columns:
                out[c] = pd.NaT
            out[c] = pd.to_datetime(out[c], errors='coerce')

        # NEW — save the 5-minute PnL stream you asked for
        if pnl_stream_rows:
            pd.DataFrame(pnl_stream_rows).to_csv(
                f"{output_folder_path}/pid_unreal_real_stream_{year}.csv",
                index=False
            )
    return out

if __name__ == "__main__":
    t0_total = perf_counter()
    counter = 0
    all_results = []
    all_option_data  = []
    all_option_data1 = []
    all_option_data2 = []
    timing_rows = []   # <-- collect structured timing rows
    log_time("🚀 Starting Rhino backtest")

    DELTA_POOL = None  

    try:
        nproc = min(4, (cpu_count() or 4))
    except Exception:
        nproc = 4
    DELTA_POOL = Pool(processes=nproc)
    log_time(f"⚙️  Multiprocessing pool started with {nproc} workers")

    # ---- Date bounds from your ranges ----
    start_date_idx = date_ranges[0][0]
    end_date_idx   = date_ranges[-1][1]

    # 📄 Read expiry mapping (once)
    mapped_days = pd.read_excel(expiry_file_path)

    # Core parsing
    mapped_days['Date']                 = pd.to_datetime(mapped_days['Date'], errors='coerce')
    mapped_days['MonthlyExpiry']        = pd.to_datetime(mapped_days['MonthlyExpiry'], errors='coerce')
    mapped_days['Next Monthly Expiry']  = pd.to_datetime(mapped_days['Next Monthly Expiry'], errors='coerce')
    mapped_days['3rd Monthly Expiry']   = pd.to_datetime(mapped_days['3rd Monthly Expiry'], errors='coerce')

    # Optional DaysToExpiry precompute
    if 'DaysToExpiry' not in mapped_days.columns:
        mapped_days['DaysToExpiry'] = (
            mapped_days['MonthlyExpiry'].dt.normalize() - mapped_days['Date'].dt.normalize()
        ).dt.days.clip(lower=0)

    mapped_days['Day'] = mapped_days['Date'].dt.day_name()
    mapped_days = mapped_days.loc[:, ~mapped_days.columns.duplicated()].copy()

    # ✅ Global date filter for mapped_days (outer bounds + specific exclusions)
    mapped_days = mapped_days[
        (mapped_days['Date'] >= pd.to_datetime(start_date_idx)) &
        (mapped_days['Date'] <= pd.to_datetime(end_date_idx)) &
        (mapped_days['Date'] != pd.Timestamp('2024-05-18')) &
        (mapped_days['Date'] != pd.Timestamp('2024-05-20'))
    ].copy()

    if mapped_days.empty:
        log_time("⚠️ No mapped days in the overall window. Nothing to do.")
        raise SystemExit(0)

    # 📊 Load index data once for the outer window
    df = pull_index_data(start_date_idx, end_date_idx, stock, mapped_days).copy()
    # Build Date/Time/Spot as required by strategy
    df['Date'] = pd.to_datetime(df.index.date)
    df['Time'] = pd.to_datetime(df.index).strftime('%H:%M')
    df['Spot'] = df['Open']
    # Ensure strategy-friendly Time normalization
    df = normalize_time_column(df)

    # 📉 Load VIX regime once (full history; the strategy picks per-day rows)
    vix_path = r"/home/newberry2/dhruvil/vix/vix_regime.csv"
    vix_df = pd.read_csv(vix_path)
    vix_df['Date'] = pd.to_datetime(vix_df['Date'], errors='coerce')

    print("mapped_days['Date'] dtype:", mapped_days['Date'].dtype)
    print("index_data['Date'] dtype:", df['Date'].dtype)

    # 📈 Iterate over each backtest sub-range
    for start_date, end_date in date_ranges:
        t0_range = perf_counter()
        counter += 1
        print(f"\n📅 Range {counter}: {start_date} → {end_date}")
        _clear_snapshot_caches()
        _clear_strike_list_cache()

        start_dt = pd.to_datetime(start_date)
        end_dt   = pd.to_datetime(end_date)

        # extend option fetch horizon (for far/next legs, rolls, etc.)
        extended_end_date = end_dt + timedelta(days=60)
        extended_end_date_str = extended_end_date.strftime("%Y-%m-%d")

        # Slice mapped_days + index for this sub-range
        md_slice = mapped_days[(mapped_days['Date'] >= start_dt) & (mapped_days['Date'] <= end_dt)].copy()
        if md_slice.empty:
            log_time("⚠️ No mapped days in this range. Skipping.")
            continue

        idx_slice = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)].copy()
        if idx_slice.empty:
            log_time("⚠️ No index rows in this range. Skipping.")
            continue

        # 🔃 Pull options data for this sub-range (with extended tail)
        t0_pull = perf_counter()
        option_data  = pull_options_data_d(start_date, extended_end_date_str, option_data_path,  stock)
        option_data1 = pull_options_data_d(start_date, extended_end_date_str, option_data_path1, stock)
        option_data2 = pull_options_data_d(start_date, extended_end_date_str, option_data_path2, stock)
        dt_pull = perf_counter() - t0_pull
        log_time(f"⏱ Options pulled in {dt_pull:.3f}s")
        timing_rows.append({
            "phase": "pull_options", "range_idx": counter,
            "start_date": start_date, "end_date": end_date, "minutes": _min(dt_pull)
        })

        # Normalize Time columns for strategy matching
        option_data  = normalize_time_column(option_data)
        option_data1 = normalize_time_column(option_data1)
        option_data2 = normalize_time_column(option_data2)

        option_data  = normalize_option_df(option_data)
        option_data1 = normalize_option_df(option_data1)
        option_data2 = normalize_option_df(option_data2)

        # Build minute indices for O(1) minute access (per range)
        def _build_minute_index(option_df):
            d = {}
            tmp = option_df.copy()
            tmp['D'] = tmp['Date'].dt.date
            for (dte, t), g in tmp.groupby(['D','Time'], sort=False):
                d[(dte, t)] = g
            return d

        MIN_IDX_MONTHLY = _build_minute_index(option_data)
        MIN_IDX_NEXT    = _build_minute_index(option_data1)
        MIN_IDX_FAR     = _build_minute_index(option_data2)

        def get_chain_minute_fast(tier, date, time_str, fallback=True, max_lookback=6):
            """
            Return chain rows for (date,time). If missing and fallback=True,
            walk back up to `max_lookback` earlier 5-min slots on the same day.
            """
            idx = {'monthly': MIN_IDX_MONTHLY, 'next': MIN_IDX_NEXT, 'far': MIN_IDX_FAR}[tier]
            d = pd.to_datetime(date).date()

            # exact hit
            df = idx.get((d, time_str))
            if df is not None and not df.empty:
                return df

            if not fallback:
                return pd.DataFrame()

            # collect times available for this date that are <= requested time
            times = sorted({t for (dd, t) in idx.keys() if dd == d and t <= time_str})

            # we only want *earlier* than the requested minute
            earlier = [t for t in times if t < time_str]
            if not earlier:
                return pd.DataFrame()

            # walk back up to max_lookback earlier stamps, newest first
            for t in reversed(earlier[-max_lookback:]):
                df = idx.get((d, t))
                if df is not None and not df.empty:
                    return df

            return pd.DataFrame()

        
        all_option_data.append(option_data)
        all_option_data1.append(option_data1)
        all_option_data2.append(option_data2)

        # 🚀 Run strategy for this range
        t0_strat = perf_counter()
        summary_df = generate_rhino_strategy(
            mapped_days=md_slice,
            option_data=option_data,
            option_data1=option_data1,
            option_data2=option_data2,
            index_df=idx_slice,
            vix_df=vix_df
        )
        dt_strat = perf_counter() - t0_strat
        log_time(f"⏱ generate_rhino_strategy in {dt_strat:.3f}s")
        timing_rows.append({
            "phase": "run_strategy", "range_idx": counter,
            "start_date": start_date, "end_date": end_date, "minutes": _min(dt_strat)
        })

        if summary_df is not None and not summary_df.empty:
            all_results.append(summary_df)
            print(f"✅ Summary rows added: {len(summary_df)}")
        else:
            print("❌ No trades/adjustments generated for this range.")

        dt_range_total = perf_counter() - t0_range
        log_time(f"⏱ Range {counter} total: {dt_range_total:.3f}s")
        timing_rows.append({
            "phase": "range_total", "range_idx": counter,
            "start_date": start_date, "end_date": end_date, "minutes": _min(dt_range_total)
        })

    # 📁 Save Final Results (once after all ranges)
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        option_data  = pd.concat(all_option_data,  ignore_index=True)
        option_data1 = pd.concat(all_option_data1, ignore_index=True)
        option_data2 = pd.concat(all_option_data2, ignore_index=True)

        # --- Rebuild FILLS etc. off the final_df ---
        fills = final_df[final_df['Action'].fillna('').str.startswith(('OPEN','EXIT','UNIVERSAL_EXIT'))].copy()

        def enrich_with_snapshots(df_in, mapped_days, option_data, option_data1, option_data2, index_df):
            """
            For every row: recompute minute-accurate leg snapshots and fill all leg fields.
            Works for OPEN / EXIT / UE and synthetic 'Current' rows.
            """
            if df_in is None or df_in.empty:
                return df_in
            out = df_in.copy()

            # (Date,Time) -> Spot
            idx = index_df.copy()
            idx['Date'] = pd.to_datetime(idx['Date']).dt.date
            spot_map = {(d, t): s for d, t, s in zip(idx['Date'], idx['Time'], idx['Spot'])}

            # Date -> mapped_days row
            md_map = {pd.to_datetime(r['Date']).date(): r for _, r in mapped_days.iterrows()}

            # ensure all leg columns exist
            fill_cols = [
                'long1_Strike','long1_Delta','long1_Price',"long1_Expiry",
                'short_Strike','short_Delta','short_Price', 'short_Expiry',
                'long2_Strike','long2_Delta','long2_Price', 'long2_Expiry',
                'cal_short_Strike','cal_short_Delta','cal_short_Price', 'cal_short_Expiry',
                'cal_long_Strike','cal_long_Delta','cal_long_Price' , 'cal_long_Expiry'
            ]
            for c in fill_cols:
                if c not in out.columns:
                    out[c] = np.nan

            for i in range(len(out)):
                idx_row = out.index[i]
                row = out.iloc[i]
                try:
                    d = pd.to_datetime(row['Date']).date()
                    t = row['Time']
                    stype = row['Type']
                    pid = row['StructureID']
                    if str(stype) not in ('rhino','calendar'):
                        continue
                    mdrow = md_map.get(d, None)
                    if mdrow is None:
                        continue
                    spot = spot_map.get((d, t), np.nan)
                    legs = _last_known_legset_for_row(row)
                    if not legs:
                        continue
                    snap = _snapshot_legs_at(d, t, pid, stype, legs, mdrow, option_data, option_data1, option_data2, spot)

                    _apply_snapshot_to_row_inplace(out, out.index[i], snap)
                    if 'Spot' in out.columns and pd.notna(spot):
                        out.at[out.index[i], 'Spot'] = float(spot)

                except Exception:
                    # leave row unchanged on any error
                    pass
            return out

        # (Optional but recommended) also enrich final_df so all action rows are fully populated too
        final_df = enrich_with_snapshots(
            final_df,
            mapped_days=mapped_days,
            option_data=option_data,
            option_data1=option_data1,
            option_data2=option_data2,
            index_df=df
        )       
        
        # 5) Re-derive fills from the *enriched* final_df (so EXIT rows have leg prices/deltas)
        fills_enriched = final_df[final_df['Action'].fillna('').str.startswith(('OPEN','EXIT','UNIVERSAL_EXIT'))].copy()

        # --- Instance numbering: per (PID, Slot) OPEN→EXIT cycle ---
        fills_enriched = fills_enriched.sort_values(['Date','Time','StructureID','Slot'], kind='mergesort').reset_index(drop=True)

        # Instance = running count of OPEN events within each (StructureID, Slot) group
        fills_enriched['Instance'] = (
            fills_enriched
            .groupby(['StructureID','Slot'], sort=False)['Action']
            .transform(lambda s: (s == 'OPEN').cumsum())
            .astype(int)
        )

        # remove any legacy "OpenPositions" / "Current" artifacts if present
        for col in ['OpenPositions']:
            if col in fills_enriched.columns:
                fills_enriched.drop(columns=[col], inplace=True)

        fills_with_open = enrich_with_snapshots(
            fills_enriched,
            mapped_days=mapped_days,
            option_data=option_data,
            option_data1=option_data1,
            option_data2=option_data2,
            index_df=df
        )
        
        def reason_from_action(a):
            a = str(a)
            if a.startswith("UNIVERSAL_EXIT"): return "UE15"
            if "short<=" in a or "Δ_short<=" in a: return "SD<=threshold"
            if "long>=" in a or "Δ_long>=" in a:  return "LD>=threshold"
            if a.startswith("TRIGGER(LD>=-0.45)"): return "LD>=-0.45_trigger"
            if a.startswith("EXIT(cal_short delta outside"): return "CAL_EXIT"
            if a == "OPEN": return "OPEN"
            if a.startswith("EXIT"): return "EXIT"
            return ""
        fills_enriched['Reason'] = fills_enriched['Action'].apply(reason_from_action)
        fills_enriched['Tier']   = fills_enriched['Comment'].str.extract(r'tier=([a-zA-Z]+)', expand=False)
        fills_enriched['OpenPositions'] = ""

        # ---- Lifecycles from ENRICHED fills ----
        rhino_lifecycle     = build_rhino_lifecycle(fills_enriched)
        calendar_lifecycle  = build_calendar_lifecycle(fills_enriched)
        combined_lifecycle  = pd.concat([rhino_lifecycle, calendar_lifecycle], ignore_index=True)

        # Sort strictly by PID then Entry Date/Time (and Instance for stability)
        combined_lifecycle = combined_lifecycle.sort_values(
            by=['PID', 'Entry Date', 'Entry Time', 'Instance'],
            kind='mergesort')

        rule_dash = (fills_enriched
                    .assign(VIX_Regime=fills_enriched['VIX_Regime'].fillna(''))
                    .groupby(['Reason','VIX_Regime','Tier'], dropna=False)
                    .size().reset_index(name='Count'))
        
        exceptions = final_df[final_df['Type']=='exception'].copy()
        
        # --- Save everything ---
        t0_save = perf_counter()
        output_base = f"{output_folder_path}/rhino_trades_summary_{year}_2"
        final_df_sorted            = sort_by_date_time_pid(final_df)
        fills_with_open_sorted     = sort_by_date_time_pid(fills_with_open)
        rule_dash_sorted           = sort_by_date_time_pid(rule_dash)         # will no-op if Date/Time not present
        exceptions_sorted          = sort_by_date_time_pid(exceptions)
        combined_lifecycle_sorted  = sort_by_date_time_pid(combined_lifecycle)
        rhino_lifecycle_sorted     = sort_by_date_time_pid(rhino_lifecycle)
        calendar_lifecycle_sorted  = sort_by_date_time_pid(calendar_lifecycle)

        final_df_sorted.to_csv(output_base + ".csv", index=False)
        fills_with_open_sorted.to_csv(output_base + "_fills_openpositions.csv", index=False)
        rule_dash_sorted.to_csv(output_base + "_rule_dashboard.csv", index=False)
        exceptions_sorted.to_csv(output_base + "_exceptions.csv", index=False)
        combined_lifecycle_sorted.to_csv(output_base + "_lifecycle_per_leg.csv", index=False)
        rhino_lifecycle_sorted.to_csv(output_base + "_rhino_lifecycle.csv", index=False)
        calendar_lifecycle_sorted.to_csv(output_base + "_calendar_lifecycle.csv", index=False)        
        
        dt_save = perf_counter() - t0_save
        log_time(f"\n✅ All summaries saved to: {output_base}*.csv")
        log_time(f"⏱ Save phase in {dt_save:.3f}s")
        timing_rows.append({
            "phase": "save_phase", "range_idx": None,
            "start_date": date_ranges[0][0], "end_date": date_ranges[-1][1], "minutes": _min(dt_save)
        })

        # ======================= PID live PnL & Drawdown tracker =======================
        # ...
        
    # Save timing rows always (even if no trades)
    try:
        timing_df = pd.DataFrame(timing_rows)
        timing_df.to_csv(f"{output_folder_path}/rhino_timings_{year}_2.csv", index=False)
        log_time(f"🧾 Timings CSV -> {output_folder_path}/rhino_timings_{year}_2.csv")
    except Exception as e:
        log_time(f"[timings save] failed: {e}")

    t1_total = perf_counter()
    total_sec = t1_total - t0_total
    log_time(f"⏱ Total runtime: {total_sec:.2f}s")
    timing_rows.append({
        "phase": "total_runtime", "range_idx": None,
        "start_date": date_ranges[0][0], "end_date": date_ranges[-1][1],
        "minutes": _min(total_sec)
    })
        # --- graceful pool shutdown ---
    try:
        if DELTA_POOL is not None:
            DELTA_POOL.close()
            DELTA_POOL.join()
            DELTA_POOL = None
            log_time("🛑 Multiprocessing pool closed")
    except Exception as e:
        log_time(f"[pool close] {e}")
