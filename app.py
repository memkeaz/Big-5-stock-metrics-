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
      /* Valuation cards */
      .vrow { display:flex; gap:1rem; flex-wrap:wrap; }
      .vbox { flex:1 1 360px; padding:16px; border-radius:14px; background:#ffffff; border:1px solid #e5e7eb; }
      .vbox.rule1 { background: #f0f7ff; border-color:#bfdbfe; }
      .vbox.dcf   { background: #f1fff7; border-color:#bbf7d0; }
      .vtitle { font-weight:700; margin-bottom:6px; color:#111827; }
      .fv { font-size:1.25rem; font-weight:800; color:#111827; margin:6px 0 2px 0; }
      .mosline { font-size:1.05rem; color:#374151; margin-top:2px; }
      .asof { color:#9ca3af; font-size:0.85rem; margin-top:4px; }
      .searchbar { display:flex; gap:8px; align-items:center; }
      .hint { color:#6b7280; font-size:0.9rem; }
      .sep { height:1px; background:#e5e7eb; margin:10px 0 14px 0; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Phil Town Big 5 Screener — Alpha-Only")
st.caption("Multi-key Alpha rotation • Big 5 guaranteed • 10/5/3/1 breakdowns • Rule #1 & DCF valuations")

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
    # unique
    seen, uniq = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k); uniq.append(k)
    return uniq

ALPHA_KEYS = _gather_alpha_keys()
if not ALPHA_KEYS:
    st.error("No Alpha Vantage keys found in Secrets. Add at least one (preferably several).")
    st.stop()

if "av_key_index" not in st.session_state:
    st.session_state.av_key_index = 0

# ================== Conservative defaults ==================
def sset(k, v):
    if k not in st.session_state: st.session_state[k] = v

# Rule #1 (Phil Town style)
sset("discount", 0.10)           # 10% MARR
sset("years_eps", 10)
sset("growth_eps_user", 0.10)    # 10% (capped ≤15% after taking min with EPS 10y CAGR)
sset("auto_pe", True)
sset("terminal_pe_man", 15.0)    # conservative cap

# DCF (Buffett-ish conservative)
sset("years_dcf", 10)
sset("growth_fcf", 0.10)         # 10% YoY FCF
sset("terminal_g", 0.02)         # 2% terminal growth

# Per-card MOS sliders (cleaner UX)
sset("mos_pct_rule1", 50)
sset("mos_pct_dcf", 50)

# App state
sset("last_query", {"query": "Uber"})
sset("data", None)               # (df, years, source, price, price_ts)

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

def series_cagr_gap(s):
    y = s.dropna()
    if len(y) < 2: return np.nan
    return cagr_over_years(y.iloc[0], y.iloc[-1], int(y.index[0]), int(y.index[-1]))

def yoy(series):
    y = series.dropna()
    if len(y) < 2: return np.nan
    prev, last = y.iloc[-2], y.iloc[-1]
    if prev <= 0 or pd.isna(prev) or pd.isna(last): return np.nan
    return (last / prev) - 1

def safe_mean(s):
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s.mean() if len(s) else np.nan

def pct(x): return "—" if pd.isna(x) else f"{x*100:.1f}%"

def normalize_capex(capex):
    if capex is None or capex.empty: return capex
    return capex.apply(lambda v: abs(v) if not pd.isna(v) else v)

def latest_positive(series, lookback=10):
    s = series.dropna()
    if lookback: s = s.iloc[-lookback:]
    s = s[s > 0]
    return s.iloc[-1] if len(s) else np.nan

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

# ---- robust getters for annual series (with fallbacks) ----
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

def _sum_series(reports, fields):
    total = None
    for f in fields:
        s = _series_from_reports(reports, f)
        total = s if total is None else total.add(s, fill_value=0)
    return total if total is not None else pd.Series(dtype="float64")

def fetch_once(symbol, apikey):
    inc = av_get("INCOME_STATEMENT", apikey, symbol=symbol).get("annualReports", [])
    bal = av_get("BALANCE_SHEET",    apikey, symbol=symbol).get("annualReports", [])
    cfs = av_get("CASH_FLOW",        apikey, symbol=symbol).get("annualReports", [])

    # Income statement
    total_revenue   = _series_from_reports(inc, "totalRevenue")
    gross_profit    = _series_from_reports(inc, "grossProfit")
    total_opex      = _series_from_reports(inc, "totalOperatingExpenses")
    net_income      = _series_from_reports(inc, "netIncome")
    diluted_eps     = _series_from_reports(inc, "dilutedEPS")
    ebit            = _series_from_reports(inc, "ebit")
    oper_income     = _series_from_reports(inc, "operatingIncome")
    tax_expense     = _series_from_reports(inc, "incomeTaxExpense")
    pretax_income   = _series_from_reports(inc, "incomeBeforeTax")

    # Balance sheet
    shares_bs       = _series_from_reports(bal, "commonStockSharesOutstanding")
    total_equity    = _series_from_reports(bal, "totalShareholderEquity")
    total_assets    = _series_from_reports(bal, "totalAssets")
    total_liab      = _series_from_reports(bal, "totalLiabilities")
    short_debt      = _series_from_reports(bal, "shortTermDebt")
    long_debt       = _series_from_reports(bal, "longTermDebt")
    total_debt      = _series_from_reports(bal, "totalDebt")
    cash            = _series_from_reports(bal, "cashAndCashEquivalentsAtCarryingValue")
    if cash.empty: cash = _series_from_reports(bal, "cashAndCashEquivalents")
    if cash.empty: cash = _series_from_reports(bal, "cashAndShortTermInvestments")

    # Cash flow
    cfo             = _series_from_reports(cfs, "operatingCashflow")
    capex           = normalize_capex(_series_from_reports(cfs, "capitalExpenditures"))
    dep_amort       = _series_from_reports(cfs, "depreciationAndAmortization")
    chg_wc          = _series_from_reports(cfs, "changeInWorkingCapital")
    if chg_wc.empty:
        chg_wc = _series_from_reports(cfs, "changeInOperatingAssetsAndLiabilities")

    # ---- Compose guaranteed series with fallbacks ----
    years = sorted(set(total_revenue.index) | set(gross_profit.index) | set(total_opex.index) |
                   set(net_income.index) | set(diluted_eps.index) | set(ebit.index) | set(oper_income.index) |
                   set(tax_expense.index) | set(pretax_income.index) | set(shares_bs.index) | set(total_equity.index) |
                   set(total_assets.index) | set(total_liab.index) | set(short_debt.index) | set(long_debt.index) |
                   set(total_debt.index) | set(cash.index) | set(cfo.index) | set(capex.index) | set(dep_amort.index) |
                   set(chg_wc.index))[-11:]
    if not years: raise RuntimeError("Alpha returned no annual years for this symbol.")

    def A(s): return s.reindex(years).astype("float64")

    total_revenue, gross_profit, total_opex, net_income, diluted_eps, ebit, oper_income, tax_expense, pretax_income,\
    shares_bs, total_equity, total_assets, total_liab, short_debt, long_debt, total_debt, cash, cfo, capex, dep_amort, chg_wc = [
        A(x) for x in [total_revenue, gross_profit, total_opex, net_income, diluted_eps, ebit, oper_income, tax_expense, pretax_income,
                       shares_bs, total_equity, total_assets, total_liab, short_debt, long_debt, total_debt, cash, cfo, capex, dep_amort, chg_wc]
    ]

    # Revenue => totalRevenue or (grossProfit + totalOperatingExpenses)
    revenue = total_revenue.copy()
    if revenue.isna().all() and not (gross_profit.isna().all() or total_opex.isna().all()):
        revenue = (gross_profit.fillna(0) + total_opex.fillna(0)).replace(0, np.nan)

    # Shares proxy
    shares = shares_bs.copy()

    # EPS => dilutedEPS or (netIncome / shares)
    eps = diluted_eps.copy()
    if eps.isna().all() and not (net_income.isna().all() or shares.isna().all()):
        with np.errstate(invalid="ignore", divide="ignore"):
            eps = net_income / shares.replace({0: np.nan})

    # Equity => totalShareholderEquity or (totalAssets - totalLiabilities)
    equity = total_equity.copy()
    if equity.isna().all() and not (total_assets.isna().all() or total_liab.isna().all()):
        equity = (total_assets - total_liab)

    # Debt => short+long or totalDebt
    debt = (short_debt.fillna(0) + long_debt.fillna(0))
    if debt.isna().all() or (debt == 0).all():
        debt = total_debt

    # FCF => CFO - |CapEx|  or  (NI + D&A − ΔWC − CapEx)
    fcf = (cfo - capex) if (not cfo.isna().all() and not capex.isna().all()) else pd.Series([np.nan]*len(years), index=years)
    if fcf.isna().all() and not net_income.isna().all() and not dep_amort.isna().all() and not chg_wc.isna().all() and not capex.isna().all():
        fcf = (net_income.fillna(0) + dep_amort.fillna(0) - chg_wc.fillna(0) - capex.fillna(0)).replace(0, np.nan)

    # ROIC => NOPAT / avg Invested Capital
    with np.errstate(divide="ignore", invalid="ignore"):
        tax_rate = (tax_expense / pretax_income).clip(0, 1)
    # NOPAT fallbacks
    if not ebit.isna().all():
        nopat = ebit * (1 - tax_rate.fillna(0.21))
    elif not oper_income.isna().all():
        nopat = oper_income * (1 - tax_rate.fillna(0.21))
    else:
        nopat = net_income  # last resort
    # Invested Capital fallbacks: Debt + Equity − Cash; if missing, approximate (Assets − Liab + Debt − Cash)
    invested_capital = (debt.fillna(0) + equity.fillna(0) - cash.fillna(0))
    if invested_capital.isna().all() or (invested_capital == 0).all():
        invested_capital = (total_assets.fillna(0) - total_liab.fillna(0) + debt.fillna(0) - cash.fillna(0))
    invested_capital = invested_capital.replace(0, np.nan)
    invested_capital_avg = (invested_capital + invested_capital.shift(1)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        roic = (nopat / invested_capital_avg).replace([np.inf, -np.inf], np.nan)

    # Guarantee: all Big-5 present (>=2 points for CAGRs/means)
    missing = []
    if revenue.dropna().shape[0] < 2: missing.append("Revenue")
    if eps.dropna().shape[0] < 2: missing.append("EPS")
    if equity.dropna().shape[0] < 2: missing.append("Equity")
    if fcf.dropna().shape[0] < 2: missing.append("FCF")
    if roic.dropna().shape[0] < 2: missing.append("ROIC")
    if missing:
        raise RuntimeError(f"Could not compute after fallbacks: {', '.join(missing)}. Alpha did not return enough usable inputs.")

    df = pd.DataFrame({
        "Revenue": revenue, "NetIncome": net_income, "EPS": eps, "Equity": equity,
        "FCF": fcf, "ROIC": roic, "SharesDiluted": shares, "Cash": cash
    }).sort_index().tail(11)

    return df, years

def get_price_intraday(symbol, apikey):
    j = av_get("TIME_SERIES_INTRADAY", apikey, symbol=symbol, interval="1min", outputsize="compact")
    ts = j.get("Time Series (1min)", {}) or j.get("Time Series (5min)", {})
    if not ts: raise RuntimeError("No intraday data")
    latest_ts = max(ts.keys())
    price = float(ts[latest_ts].get("4. close"))
    return price, latest_ts

def get_price_global(symbol, apikey):
    j = av_get("GLOBAL_QUOTE", apikey, symbol=symbol)
    price = float(j.get("Global Quote", {}).get("05. price", "nan"))
    ts = j.get("Global Quote", {}).get("07. latest trading day", "")
    return price, ts

def fetch_with_rotation(symbol):
    n = len(ALPHA_KEYS)
    start = rotate_start()
    last_err = None
    for i in range(n):
        idx = (start + i) % n
        key = ALPHA_KEYS[idx]
        try:
            df, years = fetch_once(symbol, key)
            # price
            try:
                price, pts = get_price_intraday(symbol, key)
            except AlphaRateLimit:
                price, pts = np.nan, ""
                for j in range(1, n):
                    k2 = ALPHA_KEYS[(idx + j) % n]
                    try:
                        price, pts = get_price_intraday(symbol, k2); break
                    except AlphaRateLimit: continue
                    except Exception: continue
                if pd.isna(price):
                    price, pts = get_price_global(symbol, key)
            except Exception:
                try:
                    price, pts = get_price_global(symbol, key)
                except AlphaRateLimit:
                    price, pts = np.nan, ""
                    for j in range(1, n):
                        k2 = ALPHA_KEYS[(idx + j) % n]
                        try:
                            price, pts = get_price_global(symbol, k2); break
                        except AlphaRateLimit: continue
                        except Exception: continue
            st.session_state.av_key_index = idx
            return df, years, f"Alpha (key #{idx+1})", price, pts
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

# Suggestions for company-name searches
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
        df, years, source, price, pts = fetch_with_rotation(target.upper())
    except Exception as e:
        st.error(f"Fetch error: {e}"); st.stop()
    st.session_state["data"] = (df, years, source, price, pts)

# ================== Valuation Settings (compact) ==================
with st.expander("Valuation Settings (tweak if needed)", expanded=False):
    c1, c2, c3 = st.columns(3)
    st.session_state["discount"]   = c1.number_input("Discount/MARR %", 4.0, 20.0, float(st.session_state["discount"]*100), 0.5) / 100.0
    st.session_state["years_eps"]  = c2.slider("Years (Rule #1)", 5, 15, int(st.session_state["years_eps"]), 1)
    st.session_state["years_dcf"]  = c3.slider("Years (DCF)", 5, 15, int(st.session_state["years_dcf"]), 1)

    d1, d2, d3 = st.columns(3)
    st.session_state["growth_eps_user"] = d1.number_input("Your EPS growth % (cap 15)", 0.0, 30.0, float(st.session_state["growth_eps_user"]*100), 0.5) / 100.0
    st.session_state["auto_pe"]         = d2.checkbox("Auto P/E = min(2×growth, current P/E, 15)", value=bool(st.session_state["auto_pe"]))
    st.session_state["terminal_pe_man"] = d3.number_input("Manual P/E (if Auto off)", 5.0, 30.0, float(st.session_state["terminal_pe_man"]), 0.5)

    e1, e2 = st.columns(2)
    st.session_state["growth_fcf"] = e1.number_input("FCF growth %", 0.0, 30.0, float(st.session_state["growth_fcf"]*100), 0.5) / 100.0
    st.session_state["terminal_g"] = e2.number_input("Terminal growth %", 0.0, 5.0, float(st.session_state["terminal_g"]*100), 0.25) / 100.0

# ================== Render (requires data) ==================
if st.session_state["data"] is None:
    st.info("Type a ticker or company name, pick a suggestion if shown, then click **Search**.")
    st.stop()

df, years, source, current_price, price_ts = st.session_state["data"]

# -------- Top bar: price, years, as-of --------
c1, c2, c3 = st.columns([1,1,2])
with c1:
    st.markdown("<div class='card'><div class='title'>Current Price</div>"
                f"<div class='value'>{'—' if pd.isna(current_price) else f'${current_price:,.2f}'}</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='card'><div class='title'>Years Loaded</div>"
                f"<div class='value'>{len(df.index)}</div></div>", unsafe_allow_html=True)
with c3:
    ts_text = ""
    if isinstance(price_ts, str) and price_ts:
        try:
            if " " in price_ts:
                dt = datetime.strptime(price_ts, "%Y-%m-%d %H:%M:%S")
                ts_text = dt.strftime("%b %d, %Y %I:%M %p")
            else:
                ts_text = price_ts
        except Exception:
            ts_text = price_ts
    st.markdown("<div class='card'><div class='title'>Provider</div>"
                f"<div class='value'>{source}</div>"
                f"<div class='asof'>{('As of ' + ts_text) if ts_text else ''}</div></div>", unsafe_allow_html=True)

# ================== Tabs ==================
tabs = st.tabs(["Overview", "Big 5", "Breakdowns", "Valuation"])

# -------- Overview --------
def pct_fmt(x): return "—" if pd.isna(x) else f"{x*100:.1f}%"
with tabs[0]:
    st.subheader(st.session_state["last_query"]["query"].upper())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest EPS (best)", "—" if df['EPS'].dropna().empty else f"{df['EPS'].dropna().iloc[-1]:.2f}")
    k2.metric("Latest ROIC",       "—" if df['ROIC'].dropna().empty else pct_fmt(df['ROIC'].dropna().iloc[-1]))
    k3.metric("Revenue YoY",       "—" if df['Revenue'].dropna().shape[0]<2 else pct_fmt(yoy(df["Revenue"])))
    k4.metric("FCF YoY",           "—" if df['FCF'].dropna().shape[0]<2 else pct_fmt(yoy(df["FCF"])))
    with st.expander("Mini charts"):
        cc1, cc2, cc3 = st.columns(3)
        cc1.line_chart(df[["Revenue","FCF"]].dropna(), height=200, use_container_width=True)
        cc2.line_chart(df[["EPS"]].dropna(), height=200, use_container_width=True)
        cc3.line_chart(df[["ROIC"]].dropna(), height=200, use_container_width=True)

# -------- Big 5 (guaranteed or error) --------
with tabs[1]:
    st.subheader("Big 5 — 10-Year Check (guaranteed)")
    sales_cagr_10 = series_cagr_gap(df["Revenue"])
    eps_cagr_10   = series_cagr_gap(df["EPS"])
    eqty_cagr_10  = series_cagr_gap(df["Equity"])
    fcf_cagr_10   = series_cagr_gap(df["FCF"])
    roic_avg_10   = safe_mean(df["ROIC"])
    # hard fail if any is NaN (should not happen with our fallbacks)
    if any(pd.isna(x) for x in [sales_cagr_10, eps_cagr_10, eqty_cagr_10, fcf_cagr_10, roic_avg_10]):
        st.error("A Big-5 metric could not be computed after all fallbacks. Try again later.")
        st.stop()
    def pf(v): return "PASS ✅" if v >= 0.10 else "FAIL ❌"
    big5 = pd.DataFrame({
        "Metric": ["Sales (Revenue) CAGR","EPS CAGR","Equity CAGR","FCF CAGR","ROIC (10-yr Avg)"],
        "Value (10y)":  [f"{sales_cagr_10*100:.1f}%", f"{eps_cagr_10*100:.1f}%", f"{eqty_cagr_10*100:.1f}%", f"{fcf_cagr_10*100:.1f}%", f"{roic_avg_10*100:.1f}%"],
        "Pass ≥10%?": [pf(sales_cagr_10), pf(eps_cagr_10), pf(eqty_cagr_10), pf(fcf_cagr_10), pf(roic_avg_10)]
    })
    st.dataframe(big5, use_container_width=True, height=240)

# -------- Breakdowns --------
with tabs[2]:
    st.subheader("Trends: 10 / First-5 / Last-3 / Last-1")
    def breakdown_growth(s):
        s = s.dropna()
        ten = series_cagr_gap(s) if len(s) >= 2 else np.nan
        first5 = cagr_over_years(s.iloc[0], s.iloc[min(4, len(s)-1)], int(s.index[0]), int(s.index[min(4, len(s)-1)])) if len(s) >= 5 else np.nan
        last3  = cagr_over_years(s.iloc[-4], s.iloc[-1], int(s.index[-4]), int(s.index[-1])) if len(s) >= 4 else np.nan
        last1  = yoy(s) if len(s) >= 2 else np.nan
        return ten, first5, last3, last1
    def breakdown_roic(s):
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        return (safe_mean(s) if len(s) else np.nan,
                safe_mean(s.iloc[:5]) if len(s)>=5 else np.nan,
                safe_mean(s.iloc[-3:]) if len(s)>=3 else np.nan,
                s.iloc[-1] if len(s) else np.nan)
    metrics = {
        "Sales CAGR": breakdown_growth(df["Revenue"]),
        "EPS CAGR":   breakdown_growth(df["EPS"]),
        "Equity CAGR":breakdown_growth(df["Equity"]),
        "FCF CAGR":   breakdown_growth(df["FCF"]),
        "ROIC":       breakdown_roic(df["ROIC"]),
    }
    bdf = pd.DataFrame([(k,)+v for k,v in metrics.items()], columns=["Metric","10yr","First 5yr","Last 3yr","Last 1yr"])
    for col in ["10yr","First 5yr","Last 3yr","Last 1yr"]:
        bdf[col] = bdf[col].apply(lambda x: "—" if pd.isna(x) else f"{x*100:.1f}%")
    st.dataframe(bdf, use_container_width=True, height=260)

# -------- Valuation (two separate, color-coded cards; Fair then MOS with per-card slider) --------
with tabs[3]:
    st.subheader("Intrinsic Value")

    discount = st.session_state["discount"]

    # ============== Rule #1 (EPS) ==============
    eps_series   = df["EPS"].astype(float)
    last_eps     = latest_positive(eps_series, lookback=10)
    eps_hist_cagr = series_cagr_gap(eps_series)

    candidates = [g for g in [st.session_state["growth_eps_user"], eps_hist_cagr] if pd.notna(g) and g >= 0]
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

    # ============== DCF (FCF per share) ==============
    shares_last = latest_positive(df["SharesDiluted"], lookback=10)
    fcf_last    = latest_positive(df["FCF"], lookback=10)
    fcf_ps_last = (fcf_last / shares_last) if (pd.notna(fcf_last) and pd.notna(shares_last) and shares_last > 0) else np.nan

    def intrinsic_dcf_fcf_ps(fps_last, growth, years, terminal_g, disc):
        if pd.isna(fps_last) or fps_last <= 0 or disc <= terminal_g: return np.nan
        pv = 0.0
        f = fps_last
        for t in range(1, years + 1):
            f *= (1 + growth)
            pv += f / ((1 + disc) ** t)
        f_next = f * (1 + terminal_g)
        tv = f_next / (disc - terminal_g)
        pv += tv / ((1 + disc) ** years)
        return pv

    fair_dcf = intrinsic_dcf_fcf_ps(fcf_ps_last, st.session_state["growth_fcf"], st.session_state["years_dcf"], st.session_state["terminal_g"], discount)

    # ============== UI: Two clean cards ==============
    st.markdown("<div class='vrow'>", unsafe_allow_html=True)

    # RULE 1 CARD
    st.markdown("<div class='vbox rule1'>", unsafe_allow_html=True)
    st.markdown("<div class='vtitle'>Rule #1 (EPS-based)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='fv'>Fair Value / share: {'—' if pd.isna(fair_rule1) else f'${fair_rule1:,.2f}'}</div>", unsafe_allow_html=True)
    st.session_state["mos_pct_rule1"] = st.slider("MOS % (Rule #1)", 1, 100, int(st.session_state["mos_pct_rule1"]), 1, key="mos_r1_slider")
    mos_r1 = None if pd.isna(fair_rule1) else fair_rule1 * (1 - st.session_state["mos_pct_rule1"]/100.0)
    st.markdown(
        f"<div class='mosline'>MOS Price: {'—' if mos_r1 is None else f'${mos_r1:,.2f}'} "
        f"<span class='pill'>{100 - int(st.session_state['mos_pct_rule1'])}% of Fair</span></div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # DCF CARD
    st.markdown("<div class='vbox dcf'>", unsafe_allow_html=True)
    st.markdown("<div class='vtitle'>DCF (FCF per share)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='fv'>Fair Value / share: {'—' if pd.isna(fair_dcf) else f'${fair_dcf:,.2f}'}</div>", unsafe_allow_html=True)
    st.session_state["mos_pct_dcf"] = st.slider("MOS % (DCF)", 1, 100, int(st.session_state["mos_pct_dcf"]), 1, key="mos_dcf_slider")
    mos_dcf = None if pd.isna(fair_dcf) else fair_dcf * (1 - st.session_state["mos_pct_dcf"]/100.0)
    st.markdown(
        f"<div class='mosline'>MOS Price: {'—' if mos_dcf is None else f'${mos_dcf:,.2f}'} "
        f"<span class='pill'>{100 - int(st.session_state['mos_pct_dcf'])}% of Fair</span></div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
