# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime

# ================== Page & Style ==================
st.set_page_config(page_title="Phil Town Big 5 Screener", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      .block-container { padding-top: 0.8rem; padding-bottom: 3rem; max-width: 1100px; }
      @media (max-width: 640px) { .block-container { padding-left: 0.8rem; padding-right: 0.8rem; } }
      .card { padding: 0.9rem 1rem; border-radius: 0.9rem; background: #0b1220; color: #fff; }
      .card .title { font-size: 0.9rem; opacity: 0.8; margin-bottom: 0.2rem; }
      .card .value { font-size: 1.6rem; font-weight: 700; }
      .small { color:#666; font-size:0.9rem; }
      .muted { color:#888; }
      .pill { display:inline-block; padding:2px 8px; border-radius:999px; background:#eef2ff; color:#1e40af; font-size:0.8rem; }
      .subtle { opacity: 0.8; }
      .vrow { display:flex; gap:1rem; flex-wrap:wrap; }
      .vbox { flex:1 1 320px; border:1px solid #e5e7eb; padding:14px; border-radius:12px; background:#fff; }
      .vbox.rule1 { background:#f0f7ff; border-color:#bfdbfe; }
      .vbox.dcf   { background:#f1fff7; border-color:#bbf7d0; }
      .vlabel { font-size:0.95rem; color:#374151; margin-bottom:8px; font-weight:600; }
      .fv { font-size:1.2rem; font-weight:700; }
      .mos { font-size:1.05rem; color:#374151; }
      .searchbar { display:flex; gap:8px; align-items:center; }
      .hint { color:#6b7280; font-size:0.9rem; }
      .sep { height:1px; background:#e5e7eb; margin:10px 0 14px 0; }
      .sticky-cta { margin-top: 1rem; padding-top: 0.5rem; border-top: 1px solid #e5e7eb; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Phil Town Big 5 Screener — Alpha-Only")
st.caption("Multi-key Alpha rotation · Big 5 (Strict/Turnaround) · 10/5/3/1 breakdowns · Rule #1 & DCF valuations with one MOS slider")

# ================== Gather Alpha keys & rotation ==================
def _gather_alpha_keys():
    keys = []
    for k in ["ALPHAVANTAGE_API_KEY", "ALPHAVANTAGE_API_KEY_2", "ALPHAVANTAGE_API_KEY_3"]:
        v = st.secrets.get(k, "")
        if isinstance(v, str) and v.strip():
            keys.append(v.strip())
    arr = st.secrets.get("ALPHA_KEYS", [])
    if isinstance(arr, list):
        for v in arr:
            if isinstance(v, str) and v.strip():
                keys.append(v.strip())
    csv = st.secrets.get("ALPHA_KEYS_CSV", "")
    if isinstance(csv, str) and csv.strip():
        keys.extend([x.strip() for x in csv.split(",") if x.strip()])
    seen, uniq = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k); uniq.append(k)
    return uniq

ALPHA_KEYS = _gather_alpha_keys()
if not ALPHA_KEYS:
    st.error("No Alpha Vantage keys found in Secrets. Add at least one key (preferably several).")
    st.stop()

if "av_key_index" not in st.session_state:
    st.session_state.av_key_index = 0  # active key index

# ================== Conservative defaults ==================
def sset(k, v):
    if k not in st.session_state: st.session_state[k] = v

# Rule #1 (Phil Town style)
sset("discount", 0.10)           # 10% MARR
sset("years_eps", 10)            # EPS projection years
sset("growth_eps_user", 0.10)    # user EPS growth (will be min(user, actual), capped 15%)
sset("auto_pe", True)
sset("terminal_pe_man", 15.0)    # if Auto off

# DCF (Buffett-ish conservative)
sset("years_dcf", 10)            # DCF projection years
sset("growth_fcf", 0.10)         # 10% FCF growth
sset("terminal_g", 0.02)         # 2% terminal growth

# MOS — one slider for both models
sset("mos_pct", 50)

# App state
sset("last_query", {"query": "ADBE"})
sset("data", None)               # (df, years, source, current_price)

# ================== Helpers ==================
def looks_like_ticker(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z.\-]{1,6}", s.strip()))

def cagr_over_years(first_val, last_val, first_year, last_year):
    try:
        years = last_year - first_year
        if years <= 0 or first_val <= 0 or last_val <= 0 or pd.isna(first_val) or pd.isna(last_val):
            return np.nan
        return (last_val / first_val) ** (1 / years) - 1
    except Exception:
        return np.nan

def latest_positive(series, lookback=10):
    s = series.dropna()
    if lookback: s = s.iloc[-lookback:]
    s = s[s > 0]
    return s.iloc[-1] if len(s) else np.nan

# ---- Strict vs Turnaround CAGR helpers ----
def first_positive_subseries(s: pd.Series) -> pd.Series:
    y = s.dropna()
    if y.empty: return y
    pos_idx = y[y > 0].index
    if len(pos_idx) == 0:
        return pd.Series(dtype="float64")
    first_pos_year = pos_idx[0]
    return y.loc[first_pos_year:]

def cagr_strict(s: pd.Series) -> float:
    """Requires positive start and end; else NaN."""
    y = s.dropna()
    if len(y) < 2: return np.nan
    first, first_y = float(y.iloc[0]), int(y.index[0])
    last,  last_y  = float(y.iloc[-1]), int(y.index[-1])
    if first <= 0 or last <= 0 or last_y <= first_y:
        return np.nan
    return (last / first) ** (1 / (last_y - first_y)) - 1

def cagr_turnaround(s: pd.Series) -> float:
    """Start at first positive year; else NaN."""
    sub = first_positive_subseries(s)
    if len(sub) < 2: return np.nan
    first, first_y = float(sub.iloc[0]), int(sub.index[0])
    last,  last_y  = float(sub.iloc[-1]), int(sub.index[-1])
    if first <= 0 or last <= 0 or last_y <= first_y:
        return np.nan
    return (last / first) ** (1 / (last_y - first_y)) - 1

def yoy_signed(series: pd.Series) -> float:
    """Signed YoY % change; handles zero/negative gracefully."""
    y = series.dropna()
    if len(y) < 2: return np.nan
    prev, last = float(y.iloc[-2]), float(y.iloc[-1])
    if prev == 0:
        return np.sign(last) * (np.log(abs(last) + 1) - np.log(1))
    return (last / prev) - 1.0

def pct_or_na(x): return "N/A" if pd.isna(x) else f"{x*100:.1f}%"

# ================== Alpha Vantage (with key rotation) ==================
AV_BASE = "https://www.alphavantage.co/query"

class AlphaRateLimit(Exception): pass

def av_get(fn, apikey, **extra):
    params = {"function": fn, "apikey": apikey}
    params.update(extra)
    r = requests.get(AV_BASE, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    if isinstance(j, dict):
        if j.get("Note"):          raise AlphaRateLimit(j["Note"])
        if j.get("Information"):   raise RuntimeError(f"Alpha Vantage error: {j['Information']}")
        if j.get("Error Message"): raise RuntimeError(f"Alpha Vantage error: {j['Error Message']}")
    return j

@st.cache_data(show_spinner=False, ttl=600)
def symbol_suggest(query: str, apikey: str):
    j = av_get("SYMBOL_SEARCH", apikey, keywords=query)
    best = j.get("bestMatches", [])
    rows = []
    for m in best:
        rows.append({
            "symbol":   m.get("1. symbol","").strip(),
            "name":     m.get("2. name","").strip(),
            "region":   m.get("4. region","").strip(),
            "currency": m.get("8. currency","").strip(),
            "type":     m.get("3. type","").strip()
        })
    rows.sort(key=lambda r: (0 if (r["type"].lower()=="equity" and "united states" in r["region"].lower()) else 1))
    return rows[:10]

def rotate_start(): return st.session_state.av_key_index % len(ALPHA_KEYS)

def _series_from_reports(reports, field):
    if not reports: return pd.Series(dtype="float64")
    rows = []
    for rep in reports:
        y = pd.to_datetime(rep.get("fiscalDateEnding",""), errors="coerce").year
        if pd.isna(y): continue
        rows.append((int(y), pd.to_numeric(rep.get(field), errors="coerce")))
    if not rows: return pd.Series(dtype="float64")
    s = pd.Series(dict(rows)).sort_index()
    return s.iloc[-11:].astype("float64")

def fetch_alpha_once(symbol, apikey):
    inc = av_get("INCOME_STATEMENT", apikey, symbol=symbol).get("annualReports", [])
    bal = av_get("BALANCE_SHEET",    apikey, symbol=symbol).get("annualReports", [])
    cfs = av_get("CASH_FLOW",        apikey, symbol=symbol).get("annualReports", [])

    # Income statement (core fields)
    total_revenue  = _series_from_reports(inc, "totalRevenue")
    net_income     = _series_from_reports(inc, "netIncome")
    diluted_eps    = _series_from_reports(inc, "dilutedEPS")
    ebit           = _series_from_reports(inc, "ebit")
    oper_income    = _series_from_reports(inc, "operatingIncome")
    tax_expense    = _series_from_reports(inc, "incomeTaxExpense")
    pretax_income  = _series_from_reports(inc, "incomeBeforeTax")
    gross_profit   = _series_from_reports(inc, "grossProfit")
    total_opex     = _series_from_reports(inc, "totalOperatingExpenses")

    # Balance sheet
    shares_bs      = _series_from_reports(bal, "commonStockSharesOutstanding")
    total_equity   = _series_from_reports(bal, "totalShareholderEquity")
    total_assets   = _series_from_reports(bal, "totalAssets")
    total_liab     = _series_from_reports(bal, "totalLiabilities")
    short_debt     = _series_from_reports(bal, "shortTermDebt")
    long_debt      = _series_from_reports(bal, "longTermDebt")
    total_debt     = _series_from_reports(bal, "totalDebt")
    cash           = _series_from_reports(bal, "cashAndCashEquivalentsAtCarryingValue")
    if cash.empty: cash = _series_from_reports(bal, "cashAndCashEquivalents")
    if cash.empty: cash = _series_from_reports(bal, "cashAndShortTermInvestments")

    # Cash flow
    cfo            = _series_from_reports(cfs, "operatingCashflow")
    capex_raw      = _series_from_reports(cfs, "capitalExpenditures")
    capex          = capex_raw.apply(lambda v: abs(v) if not pd.isna(v) else v) if not capex_raw.empty else capex_raw
    dep_amort      = _series_from_reports(cfs, "depreciationAndAmortization")
    chg_wc         = _series_from_reports(cfs, "changeInWorkingCapital")
    if chg_wc.empty:
        chg_wc = _series_from_reports(cfs, "changeInOperatingAssetsAndLiabilities")

    # Years union
    years = sorted(set(total_revenue.index) | set(net_income.index) | set(diluted_eps.index) |
                   set(shares_bs.index) | set(ebit.index) | set(oper_income.index) |
                   set(tax_expense.index) | set(pretax_income.index) |
                   set(total_equity.index) | set(total_assets.index) | set(total_liab.index) |
                   set(short_debt.index) | set(long_debt.index) | set(total_debt.index) |
                   set(cash.index) | set(cfo.index) | set(capex.index) |
                   set(dep_amort.index) | set(chg_wc.index) | set(gross_profit.index) |
                   set(total_opex.index))[-11:]
    if not years:
        raise RuntimeError("Alpha Vantage returned no annual years for this symbol.")

    def A(s): return s.reindex(years).astype("float64")

    total_revenue, net_income, diluted_eps, ebit, oper_income, tax_expense, pretax_income, shares_bs, total_equity, \
    total_assets, total_liab, short_debt, long_debt, total_debt, cash, cfo, capex, dep_amort, chg_wc, gross_profit, total_opex = [
        A(x) for x in [total_revenue, net_income, diluted_eps, ebit, oper_income, tax_expense, pretax_income, shares_bs, total_equity,
                       total_assets, total_liab, short_debt, long_debt, total_debt, cash, cfo, capex, dep_amort, chg_wc, gross_profit, total_opex]
    ]

    # Build metrics with accounting fallbacks
    revenue = total_revenue.copy()
    if revenue.isna().all() and not (gross_profit.isna().all() or total_opex.isna().all()):
        revenue = (gross_profit.fillna(0) + total_opex.fillna(0)).replace(0, np.nan)

    shares = shares_bs.copy()

    eps = diluted_eps.copy()
    if eps.isna().all() and not (net_income.isna().all() or shares.isna().all()):
        with np.errstate(invalid="ignore", divide="ignore"):
            eps = net_income / shares.replace({0: np.nan})

    equity = total_equity.copy()
    if equity.isna().all() and not (total_assets.isna().all() or total_liab.isna().all()):
        equity = (total_assets - total_liab)

    debt = (short_debt.fillna(0) + long_debt.fillna(0))
    if debt.isna().all() or (debt == 0).all():
        debt = total_debt

    fcf = (cfo - capex) if (not cfo.isna().all() and not capex.isna().all()) else pd.Series([np.nan]*len(years), index=years)
    if fcf.isna().all() and not net_income.isna().all() and not dep_amort.isna().all() and not chg_wc.isna().all() and not capex.isna().all():
        fcf = (net_income.fillna(0) + dep_amort.fillna(0) - chg_wc.fillna(0) - capex.fillna(0)).replace(0, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        tax_rate = (tax_expense / pretax_income).clip(0, 1)
    if not ebit.isna().all():
        nopat = ebit * (1 - tax_rate.fillna(0.21))
    elif not oper_income.isna().all():
        nopat = oper_income * (1 - tax_rate.fillna(0.21))
    else:
        nopat = net_income

    invested_capital = (debt.fillna(0) + equity.fillna(0) - cash.fillna(0))
    if invested_capital.isna().all() or (invested_capital == 0).all():
        invested_capital = (total_assets.fillna(0) - total_liab.fillna(0) + debt.fillna(0) - cash.fillna(0))
    invested_capital = invested_capital.replace(0, np.nan)
    invested_capital_avg = (invested_capital + invested_capital.shift(1)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        roic = (nopat / invested_capital_avg).replace([np.inf, -np.inf], np.nan)

    # Ensure we have at least some EPS or FCF for valuations
    df = pd.DataFrame({
        "Revenue": revenue, "NetIncome": net_income, "EPS": eps, "Equity": equity,
        "FCF": fcf, "ROIC": roic, "SharesDiluted": shares, "Cash": cash
    }).sort_index().tail(11)

    if (df["EPS"].notna().sum() < 2) and (df["FCF"].notna().sum() < 2):
        raise RuntimeError("Insufficient EPS/FCF data from Alpha Vantage for this key.")

    return df, years, "Alpha Vantage"

def get_price_intraday(symbol, apikey):
    j = av_get("TIME_SERIES_INTRADAY", apikey, symbol=symbol, interval="1min", outputsize="compact")
    ts = j.get("Time Series (1min)", {}) or j.get("Time Series (5min)", {})
    if not ts: raise RuntimeError("No intraday data")
    latest_ts = max(ts.keys())
    price = float(ts[latest_ts].get("4. close"))
    return price  # we keep it simple; not showing timestamp now

def get_price_global(symbol, apikey):
    j = av_get("GLOBAL_QUOTE", apikey, symbol=symbol)
    return float(j.get("Global Quote", {}).get("05. price", "nan"))

def fetch_with_rotation(symbol):
    n = len(ALPHA_KEYS)
    start = rotate_start()
    last_err = None
    for i in range(n):
        idx = (start + i) % n
        key = ALPHA_KEYS[idx]
        try:
            df, years, source = fetch_alpha_once(symbol, key)
            # price
            try:
                price = get_price_intraday(symbol, key)
            except AlphaRateLimit:
                price = np.nan
                for j in range(1, n):
                    k2 = ALPHA_KEYS[(idx + j) % n]
                    try:
                        price = get_price_intraday(symbol, k2); break
                    except AlphaRateLimit: continue
                    except Exception: continue
                if pd.isna(price):
                    price = get_price_global(symbol, key)
            except Exception:
                try:
                    price = get_price_global(symbol, key)
                except AlphaRateLimit:
                    price = np.nan
                    for j in range(1, n):
                        k2 = ALPHA_KEYS[(idx + j) % n]
                        try:
                            price = get_price_global(symbol, k2); break
                        except AlphaRateLimit: continue
                        except Exception: continue
            st.session_state.av_key_index = idx
            return df, years, f"{source} (key #{idx+1})", price
        except AlphaRateLimit as e:
            last_err = e; continue
        except Exception as e:
            last_err = e; continue
    raise RuntimeError(f"All Alpha keys failed. Last error: {last_err}")

# ================== Search UI ==================
st.markdown("<div class='searchbar'>", unsafe_allow_html=True)
q = st.text_input("Search company or ticker", value=st.session_state["last_query"]["query"], placeholder="e.g., UBER or Uber Technologies").strip()
go = st.button("Search", type="primary")
st.markdown("</div>", unsafe_allow_html=True)

# Suggestions
suggest_choice = None
if q and (not looks_like_ticker(q)) and len(q) >= 2:
    try:
        suggestions = symbol_suggest(q, ALPHA_KEYS[rotate_start()])
    except Exception:
        suggestions = []
    if suggestions:
        labels = [f"{r['symbol']} — {r['name']} ({r['region']})" for r in suggestions]
        pick = st.selectbox("Did you mean:", options=["(keep my input)"] + labels, index=0)
        if pick != "(keep my input)":
            k = labels.index(pick)
            suggest_choice = suggestions[k]["symbol"]

if go:
    target = suggest_choice if suggest_choice else q
    st.session_state["last_query"] = {"query": target}
    try:
        df, years, source, current_price = fetch_with_rotation(target.upper())
    except Exception as e:
        st.error(f"Fetch error: {e}"); st.stop()
    if df.empty or len(df.index) < 3:
        st.warning("Not enough annual data returned by Alpha Vantage."); st.stop()
    st.session_state["data"] = (df, years, source, current_price)

# ================== Valuation settings (LIVE, no re-search) ==================
with st.expander("Valuation Settings (Conservative Defaults)", expanded=False):
    a1, a2, a3 = st.columns(3)
    st.session_state["discount"]   = a1.number_input("MARR / Discount rate %", 4.0, 20.0, float(st.session_state["discount"]*100), 0.5) / 100.0
    st.session_state["years_eps"]  = a2.slider("Years (Rule #1 EPS)", 5, 15, int(st.session_state["years_eps"]), 1)
    st.session_state["years_dcf"]  = a3.slider("Years (FCF DCF)", 5, 15, int(st.session_state["years_dcf"]), 1)

    b1, b2, b3 = st.columns(3)
    st.session_state["growth_eps_user"] = b1.number_input("Your EPS growth % (cap ≤ 15%)", 0.0, 50.0, float(st.session_state["growth_eps_user"]*100), 0.5) / 100.0
    st.session_state["auto_pe"]         = b2.checkbox("Auto terminal P/E = min(2×growth%, current P/E, 15)", value=bool(st.session_state["auto_pe"]))
    st.session_state["terminal_pe_man"] = b3.number_input("Terminal P/E (manual if Auto off)", 5.0, 30.0, float(st.session_state["terminal_pe_man"]), 0.5)

    st.session_state["growth_fcf"] = st.number_input("FCF growth %", 0.0, 30.0, float(st.session_state["growth_fcf"]*100), 0.5) / 100.0
    st.session_state["terminal_g"] = st.number_input("Terminal growth % (FCF)", 0.0, 5.0, float(st.session_state["terminal_g"]*100), 0.25) / 100.0
    st.session_state["mos_pct"]    = st.slider("Margin of Safety % (applies to BOTH models)", 1, 100, int(st.session_state["mos_pct"]), 1)

# ================== Render (requires data) ==================
if st.session_state["data"] is None:
    st.info("Type a ticker or company name above, pick a suggestion if shown, then click **Search**.")
    st.stop()

df, years, source, current_price = st.session_state["data"]

# -------- Top bar --------
c1, c2, c3 = st.columns([1,1,2])
with c1:
    st.markdown("<div class='card'><div class='title'>Current Price</div>"
                f"<div class='value'>{'—' if pd.isna(current_price) else f'${current_price:,.2f}'}</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='card'><div class='title'>Years Loaded</div>"
                f"<div class='value'>{len(df.index)}</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='card'><div class='title'>Provider</div>"
                f"<div class='value'>{source}</div></div>", unsafe_allow_html=True)

# ================== Tabs ==================
tabs = st.tabs(["Overview", "Big 5", "Breakdowns", "Valuation"])

# -------- Overview --------
def pct_fmt(x): return "—" if pd.isna(x) else f"{x*100:.1f}%"
with tabs[0]:
    st.subheader(st.session_state["last_query"]["query"].upper())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest EPS (best)", "—" if df['EPS'].dropna().empty else f"{df['EPS'].dropna().iloc[-1]:.2f}")
    k2.metric("Latest ROIC",       "—" if df['ROIC'].dropna().empty else pct_fmt(df['ROIC'].dropna().iloc[-1]))
    k3.metric("Revenue YoY",       "—" if df['Revenue'].dropna().shape[0]<2 else pct_fmt(yoy_signed(df["Revenue"])))
    k4.metric("FCF YoY",           "—" if df['FCF'].dropna().shape[0]<2 else pct_fmt(yoy_signed(df["FCF"])))
    with st.expander("Mini charts"):
        cc1, cc2, cc3 = st.columns(3)
        cc1.line_chart(df[["Revenue","FCF"]].dropna(), height=200, use_container_width=True)
        cc2.line_chart(df[["EPS"]].dropna(), height=200, use_container_width=True)
        cc3.line_chart(df[["ROIC"]].dropna(), height=200, use_container_width=True)

# -------- Big 5 (Strict by default, optional Turnaround) --------
with tabs[1]:
    left, right = st.columns([1, 3])
    with left:
        st.subheader("Big 5 — 10-Year Check")
    with right:
        turnaround = st.toggle("Turnaround View (start from first positive year)", value=False,
                               help="OFF = strict Phil Town/Buffett style; ON = compute growth from first positive year onward.")
    CAGR_FN = cagr_turnaround if turnaround else cagr_strict

    sales_cagr_10 = CAGR_FN(df["Revenue"])
    eps_cagr_10   = CAGR_FN(df["EPS"])
    eqty_cagr_10  = CAGR_FN(df["Equity"])
    fcf_cagr_10   = CAGR_FN(df["FCF"])
    roic_vals     = df["ROIC"].replace([np.inf, -np.inf], np.nan).dropna()
    roic_avg_10   = roic_vals.mean() if len(roic_vals) else np.nan  # strict: if no ROIC, N/A → FAIL

    def passfail(x):
        return "FAIL ❌" if pd.isna(x) else ("PASS ✅" if x >= 0.10 else "FAIL ❌")

    big5 = pd.DataFrame({
        "Metric": ["Sales (Revenue) CAGR","EPS CAGR","Equity CAGR","FCF CAGR","ROIC (10-yr Avg)"],
        "Value (mode)":  [pct_or_na(sales_cagr_10), pct_or_na(eps_cagr_10), pct_or_na(eqty_cagr_10), pct_or_na(fcf_cagr_10), pct_or_na(roic_avg_10)],
        "Pass ≥10%?":    [passfail(sales_cagr_10), passfail(eps_cagr_10), passfail(eqty_cagr_10), passfail(fcf_cagr_10), passfail(roic_avg_10)]
    })
    st.dataframe(big5, use_container_width=True, height=240)
    st.caption("Strict mode requires a positive base and end value. Turnaround View starts at the first positive year. ‘N/A’ = not measurable in the selected mode (treated as FAIL).")

# -------- Breakdowns (respect the same mode) --------
with tabs[2]:
    st.subheader("Trends: 10 / First-5 / Last-3 / Last-1")
    CAGR_FN = cagr_turnaround if 'turnaround' in locals() and turnaround else cagr_strict

    def breakdown_growth(s):
        y = s.dropna()
        if len(y) < 2:
            return (np.nan, np.nan, np.nan, np.nan)
        ten = CAGR_FN(y)
        first5_series = y.iloc[:min(5, len(y))]
        first5 = CAGR_FN(first5_series) if len(first5_series) >= 2 else np.nan
        last3_series = y.iloc[-4:] if len(y) >= 4 else y
        last3 = CAGR_FN(last3_series) if len(last3_series) >= 2 else np.nan
        last1 = yoy_signed(y)
        return ten, first5, last3, last1

    def breakdown_roic(s):
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        ten   = s.mean() if len(s) else np.nan
        first = s.iloc[:5].mean() if len(s) >= 5 else np.nan
        last3 = s.iloc[-3:].mean() if len(s) >= 3 else np.nan
        last1 = s.iloc[-1] if len(s) else np.nan
        return ten, first, last3, last1

    metrics = {
        "Sales CAGR": breakdown_growth(df["Revenue"]),
        "EPS CAGR":   breakdown_growth(df["EPS"]),
        "Equity CAGR":breakdown_growth(df["Equity"]),
        "FCF CAGR":   breakdown_growth(df["FCF"]),
        "ROIC":       breakdown_roic(df["ROIC"]),
    }
    bdf = pd.DataFrame([(k,)+v for k,v in metrics.items()],
                       columns=["Metric","10yr","First 5yr","Last 3yr","Last 1yr"])
    for col in ["10yr","First 5yr","Last 3yr","Last 1yr"]:
        bdf[col] = bdf[col].apply(pct_or_na)
    st.dataframe(bdf, use_container_width=True, height=260)
    st.caption("Breakdowns honor the same mode: Strict vs Turnaround.")

# -------- Valuation (Fair Value + MOS for both) --------
with tabs[3]:
    st.subheader("Intrinsic Value (Two Models)")

    # Inputs / sliders
    discount    = st.session_state["discount"]
    mos_frac    = st.session_state["mos_pct"] / 100.0
    percent_of_fair = (1.0 - mos_frac) * 100.0

    # ---- Rule #1 (EPS) ----
    eps_series   = df["EPS"].astype(float)
    last_eps     = latest_positive(eps_series, lookback=10)
    eps_hist_cagr = cagr_strict(eps_series)  # strict for valuation input
    candidates   = [g for g in [st.session_state["growth_eps_user"], eps_hist_cagr] if pd.notna(g) and g >= 0]
    rule1_growth = min(candidates) if candidates else np.nan
    if pd.notna(rule1_growth): rule1_growth = min(rule1_growth, 0.15)

    current_pe = np.nan
    if pd.notna(current_price) and pd.notna(last_eps) and last_eps > 0:
        current_pe = current_price / last_eps

    if st.session_state["auto_pe"] and pd.notna(rule1_growth):
        pe_from_growth = (rule1_growth * 100) * 2
        choices = [pe_from_growth]
        if pd.notna(current_pe) and current_pe > 0: choices.append(current_pe)
        term_pe = min(min(choices), 15.0)
    else:
        term_pe = st.session_state["terminal_pe_man"]

    def rule1_eps_prices(eps_now, growth, years, terminal_pe, marr):
        if pd.isna(eps_now) or eps_now <= 0 or pd.isna(growth) or pd.isna(terminal_pe) or terminal_pe <= 0:
            return np.nan, np.nan, np.nan
        fut_eps = eps_now * ((1 + growth) ** years)
        sticker = fut_eps * terminal_pe
        fair    = sticker / ((1 + marr) ** years)
        return fut_eps, sticker, fair

    _, _, fair_rule1 = rule1_eps_prices(last_eps, rule1_growth, st.session_state["years_eps"], term_pe, discount)
    mos_price_rule1  = fair_rule1 * (1 - mos_frac) if pd.notna(fair_rule1) else np.nan

    # ---- DCF (FCF per share) ----
    shares_last = latest_positive(df["SharesDiluted"], lookback=10)
    fcf_last    = latest_positive(df["FCF"], lookback=10)
    fcf_ps_last = (fcf_last / shares_last) if (pd.notna(fcf_last) and pd.notna(shares_last) and shares_last > 0) else np.nan

    def intrinsic_dcf_fcf_per_share(fps_last, growth, years, terminal_g, discount):
        if pd.isna(fps_last) or fps_last <= 0 or discount <= terminal_g: return np.nan
        pv = 0.0
        f = fps_last
        for t in range(1, years + 1):
            f *= (1 + growth)
            pv += f / ((1 + discount) ** t)
        f_next = f * (1 + terminal_g)
        tv = f_next / (discount - terminal_g)
        pv += tv / ((1 + discount) ** years)
        return pv

    iv_dcf          = intrinsic_dcf_fcf_per_share(fcf_ps_last, st.session_state["growth_fcf"], st.session_state["years_dcf"], st.session_state["terminal_g"], discount)
    mos_price_dcf   = iv_dcf * (1 - mos_frac) if pd.notna(iv_dcf) else np.nan

    # --- Display ---
    st.markdown("<div class='vrow'>", unsafe_allow_html=True)

    # Rule #1 card
    st.markdown("<div class='vbox rule1'>", unsafe_allow_html=True)
    st.markdown("<div class='vlabel'>Rule #1 (EPS)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='fv'>Fair Value / share: {'—' if pd.isna(fair_rule1) else f'${fair_rule1:,.2f}'}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='mos'>MOS Price: "
        f"{'—' if pd.isna(mos_price_rule1) else f'${mos_price_rule1:,.2f}'} "
        f"<span class='muted'>(≈ {percent_of_fair:.0f}% of Fair)</span></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='small subtle'>Growth {('—' if pd.isna(rule1_growth) else f'{rule1_growth*100:.1f}%')}, "
        f"Terminal P/E {('—' if pd.isna(term_pe) else f'{term_pe:.1f}')}, Discount {discount*100:.1f}%</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # DCF card
    st.markdown("<div class='vbox dcf'>", unsafe_allow_html=True)
    st.markdown("<div class='vlabel'>DCF (FCF per share)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='fv'>Fair Value / share: {'—' if pd.isna(iv_dcf) else f'${iv_dcf:,.2f}'}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='mos'>MOS Price: "
        f"{'—' if pd.isna(mos_price_dcf) else f'${mos_price_dcf:,.2f}'} "
        f"<span class='muted'>(≈ {percent_of_fair:.0f}% of Fair)</span></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='small subtle'>FCF growth {st.session_state['growth_fcf']*100:.1f}%, "
        f"Terminal g {st.session_state['terminal_g']*100:.1f}%, Discount {discount*100:.1f}%</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # One MOS slider for both
    st.markdown("<div class='sticky-cta'>", unsafe_allow_html=True)
    st.slider("Margin of Safety % (discount from Fair Value — controls BOTH models)", 1, 100, key="mos_pct")
    st.markdown("</div>", unsafe_allow_html=True)
