# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests

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
      .subtle { opacity: 0.8; }
      .vrow { display:flex; gap:1rem; flex-wrap:wrap; }
      .vbox { flex:1 1 320px; border:1px solid #e5e7eb; padding:14px; border-radius:12px; background:#fff; }
      .vlabel { font-size:0.95rem; color:#374151; margin-bottom:8px; font-weight:600; }
      .fv { font-size:1.2rem; font-weight:700; }
      .mos { font-size:1.05rem; color:#374151; }
      .sticky-cta { margin-top: 1rem; padding-top: 0.5rem; border-top: 1px solid #e5e7eb; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Phil Town Big 5 Screener — Alpha-Only")
st.caption("Alpha Vantage with multi-key rotation. Big 5 (10y) + breakdowns (10/5/3/1) and two valuations (Rule #1 EPS & FCF-DCF). One MOS slider updates both live.")

# ================== Collect Alpha keys & rotation ==================
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
        for v in csv.split(","):
            if v.strip():
                keys.append(v.strip())
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
    st.session_state.av_key_index = 0  # active key index

# ================== Conservative defaults ==================
def sset(k, v):
    if k not in st.session_state: st.session_state[k] = v

# Global valuation defaults (you requested)
sset("discount", 0.10)           # 10% MARR
sset("years_eps", 10)            # EPS projection years
sset("growth_eps_user", 0.10)    # 10% user EPS growth (capped to ≤15% later)
sset("auto_pe", True)
sset("terminal_pe_man", 15.0)    # if Auto off

sset("years_dcf", 10)            # DCF projection years
sset("growth_fcf", 0.10)         # 10% FCF YoY default
sset("terminal_g", 0.02)         # 2% terminal growth

sset("mos_pct", 50)              # One MOS slider for both
sset("last_query", {"ticker": "ADBE"})
sset("data", None)               # (df, years, source, current_price)

# ================== Helpers ==================
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

class AlphaRateLimit(Exception):
    pass

def av_get(fn, symbol, apikey, **extra):
    params = {"function": fn, "symbol": symbol, "apikey": apikey}
    params.update(extra)
    r = requests.get(AV_BASE, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    if isinstance(j, dict):
        if j.get("Note"):
            raise AlphaRateLimit(j["Note"])
        if j.get("Information"):
            raise RuntimeError(f"Alpha Vantage error: {j['Information']}")
        if j.get("Error Message"):
            raise RuntimeError(f"Alpha Vantage error: {j['Error Message']}")
    return j

def av_series_annual(reports, field):
    if not reports: return pd.Series(dtype="float64")
    rows = []
    for rep in reports:
        y = pd.to_datetime(rep.get("fiscalDateEnding", ""), errors="coerce").year
        if pd.isna(y): continue
        rows.append((int(y), pd.to_numeric(rep.get(field), errors="coerce")))
    if not rows: return pd.Series(dtype="float64")
    s = pd.Series(dict(rows)).sort_index()
    return s.iloc[-11:].astype("float64")

def av_series_sum(reports, fields):
    total = None
    for f in fields:
        s = av_series_annual(reports, f)
        total = s if total is None else total.add(s, fill_value=0)
    return total if total is not None else pd.Series(dtype="float64")

def fetch_alpha_once(symbol, apikey):
    inc = av_get("INCOME_STATEMENT", symbol, apikey).get("annualReports", [])
    bal = av_get("BALANCE_SHEET",    symbol, apikey).get("annualReports", [])
    cfs = av_get("CASH_FLOW",        symbol, apikey).get("annualReports", [])

    revenue        = av_series_annual(inc, "totalRevenue")
    net_income     = av_series_annual(inc, "netIncome")
    eps_named      = av_series_annual(inc, "dilutedEPS")
    ebit           = av_series_annual(inc, "ebit")
    tax_expense    = av_series_annual(inc, "incomeTaxExpense")
    pretax_income  = av_series_annual(inc, "incomeBeforeTax")

    shares_diluted = av_series_annual(bal, "commonStockSharesOutstanding")
    equity         = av_series_annual(bal, "totalShareholderEquity")
    debt_primary   = av_series_sum(bal, ["shortTermDebt", "longTermDebt"])
    debt_alt       = av_series_sum(bal, ["currentLongTermDebt", "longTermDebtNoncurrent"])
    total_debt     = debt_primary if (debt_primary is not None and not debt_primary.empty) else debt_alt
    if total_debt is None or total_debt.empty: total_debt = av_series_annual(bal, "totalDebt")
    if total_debt is None or total_debt.empty: total_debt = av_series_annual(bal, "shortLongTermDebtTotal")
    cash = av_series_annual(bal, "cashAndCashEquivalentsAtCarryingValue")
    if cash.empty: cash = av_series_annual(bal, "cashAndCashEquivalents")
    if cash.empty: cash = av_series_annual(bal, "cashAndShortTermInvestments")

    cfo   = av_series_annual(cfs, "operatingCashflow")
    capex = normalize_capex(av_series_annual(cfs, "capitalExpenditures"))

    years = sorted(set(revenue.index) | set(net_income.index) | set(eps_named.index) |
                   set(shares_diluted.index) | set(ebit.index) | set(tax_expense.index) |
                   set(pretax_income.index) | set(equity.index) | set(total_debt.index) |
                   set(cash.index) | set(cfo.index) | set(capex.index))[-11:]
    if not years:
        raise RuntimeError("Alpha Vantage returned no annual years for this symbol.")

    def A(s): return s.reindex(years).astype("float64")
    revenue, net_income, eps_named, shares_diluted, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex = [
        A(x) for x in [revenue, net_income, eps_named, shares_diluted, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex]
    ]

    # EPS best: named EPS or NI/Shares
    eps_alt = pd.Series(dtype="float64")
    if net_income.notna().any() and shares_diluted.notna().any():
        with np.errstate(invalid="ignore", divide="ignore"):
            eps_alt = (net_income / shares_diluted.replace({0: np.nan})).astype(float)
    eps_best = eps_named.fillna(eps_alt) if not eps_alt.empty else eps_named

    # FCF = CFO - |CapEx|
    fcf = (cfo - capex) if (not cfo.isna().all() and not capex.isna().all()) else pd.Series([np.nan]*len(years), index=years)

    # ROIC proxy (NOPAT / avg invested capital)
    with np.errstate(divide="ignore", invalid="ignore"):
        tax_rate = (tax_expense / pretax_income).clip(0, 1)
    nopat = (ebit * (1 - tax_rate.fillna(0.21))) if not ebit.isna().all() else net_income
    invested_capital = (total_debt.fillna(0) + equity.fillna(0) - cash.fillna(0)).replace({0: np.nan})
    invested_capital_avg = (invested_capital + invested_capital.shift(1)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        roic = (nopat / invested_capital_avg).replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame({
        "Revenue": revenue, "NetIncome": net_income, "EPS": eps_best, "Equity": equity,
        "FCF": fcf, "ROIC": roic, "SharesDiluted": shares_diluted
    }).sort_index().tail(11)

    # ensure we have some EPS or FCF
    if (df["EPS"].notna().sum() < 2) and (df["FCF"].notna().sum() < 2):
        raise RuntimeError("Insufficient EPS/FCF data from Alpha for this key.")

    return df, years, "Alpha Vantage"

def get_price_alpha_intraday(symbol, apikey):
    j = av_get("TIME_SERIES_INTRADAY", symbol, apikey, interval="1min", outputsize="compact")
    ts = j.get("Time Series (1min)", {}) or j.get("Time Series (5min)", {})
    if not ts:
        raise RuntimeError("No intraday time series returned.")
    latest_ts = max(ts.keys())
    price = float(ts[latest_ts].get("4. close"))
    return price

def get_price_alpha_global(symbol, apikey):
    j = av_get("GLOBAL_QUOTE", symbol, apikey)
    return float(j.get("Global Quote", {}).get("05. price", "nan"))

def fetch_alpha_with_rotation(symbol):
    n = len(ALPHA_KEYS)
    start = st.session_state.av_key_index % n
    last_err = None
    for i in range(n):
        idx = (start + i) % n
        key = ALPHA_KEYS[idx]
        try:
            df, years, source = fetch_alpha_once(symbol, key)
            # Try intraday price; rotate keys for price on rate-limit
            try:
                price = get_price_alpha_intraday(symbol, key)
            except AlphaRateLimit:
                price = np.nan
                for j in range(1, n):
                    k2 = ALPHA_KEYS[(idx + j) % n]
                    try:
                        price = get_price_alpha_intraday(symbol, k2); break
                    except AlphaRateLimit:
                        continue
                    except Exception:
                        continue
                if pd.isna(price):
                    price = get_price_alpha_global(symbol, key)
            except Exception:
                try:
                    price = get_price_alpha_global(symbol, key)
                except AlphaRateLimit:
                    price = np.nan
                    for j in range(1, n):
                        k2 = ALPHA_KEYS[(idx + j) % n]
                        try:
                            price = get_price_alpha_global(symbol, k2); break
                        except AlphaRateLimit:
                            continue
                        except Exception:
                            continue
            st.session_state.av_key_index = idx
            return df, years, f"{source} (key #{idx+1})", price
        except AlphaRateLimit as e:
            last_err = e; continue
        except Exception as e:
            last_err = e; continue
    raise RuntimeError(f"All Alpha keys failed. Last error: {last_err}")

# ================== Search form ==================
with st.form("search"):
    c1, c2 = st.columns([2, 1])
    with c1:
        ticker = st.text_input("Ticker", value=st.session_state["last_query"]["ticker"]).strip().upper()
    with c2:
        st.text_input("Data source", value="Alpha Vantage (multi-key rotation)", disabled=True)
    go = st.form_submit_button("Search")

if go:
    try:
        df, years, source, current_price = fetch_alpha_with_rotation(ticker)
    except Exception as e:
        st.error(f"Fetch error: {e}"); st.stop()
    if df.empty or len(df.index) < 3:
        st.warning("Not enough annual data returned by Alpha Vantage."); st.stop()
    st.session_state["data"] = (df, years, source, current_price)
    st.session_state["last_query"] = {"ticker": ticker}

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

    c1, c2, c3 = st.columns(3)
    st.session_state["growth_fcf"] = c1.number_input("FCF growth %", 0.0, 30.0, float(st.session_state["growth_fcf"]*100), 0.5) / 100.0
    st.session_state["terminal_g"] = c2.number_input("Terminal growth % (FCF)", 0.0, 5.0, float(st.session_state["terminal_g"]*100), 0.25) / 100.0
    st.session_state["mos_pct"]    = c3.slider("Margin of Safety % (applies to BOTH)", 1, 100, int(st.session_state["mos_pct"]), 1)

# ================== Render (requires data) ==================
if st.session_state["data"] is None:
    st.info("Enter a ticker and click **Search**. The app will rotate across your Alpha keys and show a fresh intraday price.")
    st.stop()

df, years, source, current_price = st.session_state["data"]

# -------- Top bar: price first --------
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
    st.subheader(f"{st.session_state['last_query']['ticker']}")
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

# -------- Big 5 --------
with tabs[1]:
    st.subheader("Big 5 — 10-Year Check")
    sales_cagr_10 = series_cagr_gap(df["Revenue"])
    eps_cagr_10   = series_cagr_gap(df["EPS"])
    eqty_cagr_10  = series_cagr_gap(df["Equity"])
    fcf_cagr_10   = series_cagr_gap(df["FCF"])
    roic_avg_10   = safe_mean(df["ROIC"])
    def pf(v): return "PASS ✅" if not pd.isna(v) and v >= 0.10 else ("—" if pd.isna(v) else "FAIL ❌")
    big5 = pd.DataFrame({
        "Metric": ["Sales (Revenue) CAGR","EPS CAGR","Equity CAGR","FCF CAGR","ROIC (10-yr Avg)"],
        "Value (10y)":  [pct_fmt(sales_cagr_10), pct_fmt(eps_cagr_10), pct_fmt(eqty_cagr_10), pct_fmt(fcf_cagr_10), pct_fmt(roic_avg_10)],
        "Pass ≥10%?": [pf(sales_cagr_10), pf(eps_cagr_10), pf(eqty_cagr_10), pf(fcf_cagr_10), pf(roic_avg_10)]
    })
    st.dataframe(big5, use_container_width=True, height=240)

# -------- Breakdowns --------
with tabs[2]:
    st.subheader("Trends: 10 / First-5 / Last-3 / Last-1")
    def breakdown_growth(s):
        s = s.dropna()
        if len(s) < 2: return np.nan, np.nan, np.nan, np.nan
        ten = series_cagr_gap(s)
        first5 = cagr_over_years(s.iloc[0], s.iloc[min(4, len(s)-1)],
                                 int(s.index[0]), int(s.index[min(4, len(s)-1)])) if len(s) >= 5 else np.nan
        last3  = cagr_over_years(s.iloc[-4], s.iloc[-1], int(s.index[-4]), int(s.index[-1])) if len(s) >= 4 else np.nan
        last1  = yoy(s)
        return ten, first5, last3, last1
    def breakdown_roic(s):
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if len(s)==0: return np.nan, np.nan, np.nan, np.nan
        return safe_mean(s), safe_mean(s.iloc[:5]) if len(s)>=5 else np.nan, safe_mean(s.iloc[-3:]) if len(s)>=3 else np.nan, s.iloc[-1]
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

# -------- Valuation (Fair Value + MOS for BOTH models) --------
with tabs[3]:
    st.subheader("Intrinsic Value (Two Models)")

    # Grab MOS as a fraction once, used everywhere; slider lives at the bottom of this tab
    mos_frac = st.session_state["mos_pct"] / 100.0
    percent_of_fair = (1.0 - mos_frac) * 100.0  # e.g., MOS 40% -> 60% of Fair

    # --- Rule #1 EPS ---
    eps_series   = df["EPS"].astype(float)
    last_eps     = latest_positive(eps_series, lookback=10)
    eps_hist_cagr = series_cagr_gap(eps_series)

    # Rule #1 growth: min(user, 10y EPS CAGR), cap 15%
    candidates = [g for g in [st.session_state["growth_eps_user"], eps_hist_cagr] if pd.notna(g) and g >= 0]
    rule1_growth = min(candidates) if candidates else np.nan
    if pd.notna(rule1_growth): rule1_growth = min(rule1_growth, 0.15)

    # Current P/E if possible
    current_pe = np.nan
    if pd.notna(current_price) and pd.notna(last_eps) and last_eps > 0:
        current_pe = current_price / last_eps

    # Terminal P/E (conservative auto: min(2×growth%, current P/E, 15))
    if st.session_state["auto_pe"] and pd.notna(rule1_growth):
        pe_from_growth = (rule1_growth * 100) * 2
        choices = [pe_from_growth]
        if pd.notna(current_pe) and current_pe > 0: choices.append(current_pe)
        term_pe = min(min(choices), 15.0)
    else:
        term_pe = st.session_state["terminal_pe_man"]

    def rule1_eps_prices(eps_now, growth, years, terminal_pe, marr):
        if pd.isna(eps_now) or eps_now <= 0: return np.nan, np.nan, np.nan
        if pd.isna(growth) or pd.isna(terminal_pe) or terminal_pe <= 0: return np.nan, np.nan, np.nan
        fut_eps = eps_now * ((1 + growth) ** years)
        sticker = fut_eps * terminal_pe
        fair    = sticker / ((1 + marr) ** years)
        return fut_eps, sticker, fair

    fut_eps, sticker, fair_rule1 = rule1_eps_prices(
        last_eps, rule1_growth, st.session_state["years_eps"], term_pe, st.session_state["discount"]
    )
    mos_price_rule1 = fair_rule1 * (1 - mos_frac) if pd.notna(fair_rule1) else np.nan

    # --- FCF-DCF per share ---
    shares_last = latest_positive(df["SharesDiluted"], lookback=10)
    fcf_last    = latest_positive(df["FCF"], lookback=10)
    fcf_ps_last = (fcf_last / shares_last) if (pd.notna(fcf_last) and pd.notna(shares_last) and shares_last > 0) else np.nan

    def intrinsic_dcf_fcf_per_share(fps_last, growth, years, terminal_g, discount):
        if pd.isna(fps_last) or fps_last <= 0: return np.nan
        if discount <= terminal_g: return np.nan
        pv = 0.0
        f = fps_last
        for t in range(1, years + 1):
            f *= (1 + growth)
            pv += f / ((1 + discount) ** t)
        f_next = f * (1 + terminal_g)
        tv = f_next / (discount - terminal_g)
        pv += tv / ((1 + discount) ** years)
        return pv

    iv_dcf  = intrinsic_dcf_fcf_per_share(
        fcf_ps_last, st.session_state["growth_fcf"], st.session_state["years_dcf"],
        st.session_state["terminal_g"], st.session_state["discount"]
    )
    mos_price_dcf = iv_dcf * (1 - mos_frac) if pd.notna(iv_dcf) else np.nan

    # --- Display: each model shows Fair Value first, then MOS Price + % of Fair ---
    st.markdown("<div class='vrow'>", unsafe_allow_html=True)

    # Rule #1 box
    st.markdown("<div class='vbox'>", unsafe_allow_html=True)
    st.markdown("<div class='vlabel'>Rule #1 (EPS)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='fv'>Fair Value / share: {'—' if pd.isna(fair_rule1) else f'${fair_rule1:,.2f}'}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='mos'>MOS Price: "
        f"{'—' if pd.isna(mos_price_rule1) else f'${mos_price_rule1:,.2f}'} "
        f"<span class='muted'>(≈ {percent_of_fair:.0f}% of Fair)</span></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='small subtle'>Growth used {('—' if pd.isna(rule1_growth) else f'{rule1_growth*100:.1f}%')}, "
        f"Terminal P/E {('—' if pd.isna(term_pe) else f'{term_pe:.1f}')}, Discount {st.session_state['discount']*100:.1f}%</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # DCF box
    st.markdown("<div class='vbox'>", unsafe_allow_html=True)
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
        f"Terminal g {st.session_state['terminal_g']*100:.1f}%, Discount {st.session_state['discount']*100:.1f}%</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # end vrow

    # Bottom MOS slider (live) — controls both models
    st.markdown("<div class='sticky-cta'>", unsafe_allow_html=True)
    st.slider("Margin of Safety % (discount from Fair Value — controls BOTH models)", 1, 100, key="mos_pct")
    st.markdown("</div>", unsafe_allow_html=True)
