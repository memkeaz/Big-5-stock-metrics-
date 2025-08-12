import streamlit as st
import pandas as pd
import numpy as np
import requests
from typing import Tuple, List

# -------------------- Page Config --------------------
st.set_page_config(page_title="Phil Town Big 5 Screener", layout="wide")
st.title("Phil Town Big 5 — 10-Year Screener")
st.caption("Checks 10-year CAGRs for Sales, EPS, Equity, FCF, and the 10-yr Avg ROIC (NOPAT / Invested Capital). Adds ROIC Breakdown + Intrinsic Value (EPS & FCF DCF).")

# -------------------- Cache Control --------------------
scol1, scol2 = st.columns([3, 1])
with scol1:
    st.info("If results don't show, click **Clear Cache** then search again.")
with scol2:
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared. Run a new search.")

# -------------------- Sidebar --------------------
st.sidebar.header("Settings")
provider = st.sidebar.radio("Data Provider", ["Alpha Vantage", "FMP"])
ticker = st.text_input("Enter ticker (e.g., AAPL, MSFT, ADBE):", value="MSFT").strip().upper()
run = st.button("Search")

AV_KEY = st.secrets.get("ALPHAVANTAGE_API_KEY", "").strip()
FMP_KEY = st.secrets.get("FMP_API_KEY", "").strip()
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "").strip()

# ---------- Valuation assumptions ----------
st.sidebar.markdown("### Valuation Assumptions")
# EPS (P/E) model
years_eps = st.sidebar.slider("Years (EPS model)", 5, 15, 10)
growth_eps = st.sidebar.number_input("EPS growth (annual, %)", 0.0, 50.0, 12.0, step=0.5) / 100.0
terminal_pe = st.sidebar.number_input("Terminal P/E", 5.0, 60.0, 20.0, step=0.5)
# FCF DCF model
years_dcf = st.sidebar.slider("Years (FCF DCF)", 5, 15, 10)
growth_fcf = st.sidebar.number_input("FCF growth (annual, %)", 0.0, 50.0, 10.0, step=0.5) / 100.0
terminal_g = st.sidebar.number_input("Terminal growth (FCF, %)", 0.0, 6.0, 3.0, step=0.25) / 100.0
# Discount rate (both)
discount = st.sidebar.number_input("Discount rate (both models, %)", 4.0, 20.0, 10.0, step=0.5) / 100.0

# -------------------- Demo Mode --------------------
def demo_msft_df():
    years = list(range(2015, 2025))
    shares = [7900, 7800, 7750, 7720, 7700, 7680, 7650, 7450, 7420, 7400]  # millions, rough demo
    df = pd.DataFrame({
        "Revenue": [93580, 85320, 89950, 110360, 125843, 143015, 168088, 198270, 211915, 245000],
        "EPS":     [2.48, 2.79, 3.31, 2.13, 5.76, 6.20, 8.05, 9.21, 9.68, 11.00],
        "Equity":  [72163, 82718, 82572, 82572, 118304, 118304, 166542, 166542, 194000, 210000],
        "FCF":     [23969, 31378, 31922, 32694, 45230, 45300, 56300, 65700, 67800, 78000],
        "ROIC":    [0.12, 0.13, 0.15, 0.10, 0.18, 0.19, 0.21, 0.22, 0.20, 0.21],
        "SharesDiluted": shares
    }, index=years).astype(float)
    return df.iloc[-10:], years[-10:]

# -------------------- Helpers --------------------
def cagr(series: pd.Series, years: int) -> float:
    try:
        if years <= 0 or series.empty: return np.nan
        first, last = series.iloc[0], series.iloc[-1]
        if pd.isna(first) or pd.isna(last) or first <= 0 or last <= 0: return np.nan
        return (last / first) ** (1 / years) - 1
    except Exception:
        return np.nan

def pct(x: float) -> str:
    return "—" if (x is None or pd.isna(x)) else f"{x*100:.1f}%"

def safe_mean(s: pd.Series) -> float:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s.mean() if len(s) else np.nan

# -------------------- Alpha Vantage fetch --------------------
AV_BASE = "https://www.alphavantage.co/query"

def av_get(fn: str, symbol: str, apikey: str):
    params = {"function": fn, "symbol": symbol, "apikey": apikey}
    r = requests.get(AV_BASE, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    return j.get("annualReports", [])

def av_series(reports, field) -> pd.Series:
    if not reports: return pd.Series(dtype="float64")
    rows = []
    for rep in reports:
        y = pd.to_datetime(rep.get("fiscalDateEnding", ""), errors="coerce").year
        if pd.isna(y): continue
        val = pd.to_numeric(rep.get(field, None), errors="coerce")
        rows.append((int(y), val))
    if not rows: return pd.Series(dtype="float64")
    s = pd.Series(dict(rows)).sort_index()
    return s.iloc[-11:].astype("float64")

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_alpha_vantage(symbol: str, apikey: str):
    inc = av_get("INCOME_STATEMENT", symbol, apikey)
    bal = av_get("BALANCE_SHEET",  symbol, apikey)
    cfs = av_get("CASH_FLOW",      symbol, apikey)

    revenue        = av_series(inc, "totalRevenue")
    net_income     = av_series(inc, "netIncome")
    diluted_eps    = av_series(inc, "dilutedEPS")
    diluted_shares = av_series(inc, "weightedAverageShsOutDil")
    ebit           = av_series(inc, "ebit")
    tax_expense    = av_series(inc, "incomeTaxExpense")
    pretax_income  = av_series(inc, "incomeBeforeTax")

    equity     = av_series(bal, "totalShareholderEquity")
    total_debt = av_series(bal, "totalDebt")
    if total_debt.empty:
        total_debt = av_series(bal, "shortLongTermDebtTotal")
    cash       = av_series(bal, "cashAndCashEquivalentsAtCarryingValue")
    if cash.empty:
        cash = av_series(bal, "cashAndShortTermInvestments")

    cfo   = av_series(cfs, "operatingCashflow")
    capex = av_series(cfs, "capitalExpenditures")

    years = sorted(set(revenue.index) | set(net_income.index) | set(diluted_eps.index) |
                   set(diluted_shares.index) | set(ebit.index) | set(tax_expense.index) |
                   set(pretax_income.index) | set(equity.index) | set(total_debt.index) |
                   set(cash.index) | set(cfo.index) | set(capex.index))[-11:]

    def A(s): return s.reindex(years).astype("float64") if years else pd.Series(dtype="float64")
    revenue, net_income, diluted_eps, diluted_shares, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex = [
        A(x) for x in [revenue, net_income, diluted_eps, diluted_shares, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex]
    ]

    # EPS fallback (NI / diluted shares)
    eps = diluted_eps.copy()
    if eps.isna().all() and not net_income.isna().all() and not diluted_shares.isna().all():
        with np.errstate(invalid="ignore", divide="ignore"):
            eps = net_income / diluted_shares.replace({0: np.nan})

    # FCF = CFO − CapEx
    fcf = (cfo - capex) if (not cfo.isna().all() and not capex.isna().all()) else pd.Series([np.nan]*len(years), index=years)

    # ROIC = NOPAT / Invested Capital
    with np.errstate(divide="ignore", invalid="ignore"):
        tax_rate = (tax_expense / pretax_income).clip(0, 1)
    nopat = (ebit * (1 - tax_rate.fillna(0.21))) if not ebit.isna().all() else net_income
    invested_capital = (total_debt.fillna(0) + equity.fillna(0) - cash.fillna(0)).replace({0: np.nan})
    with np.errstate(divide="ignore", invalid="ignore"):
        roic = (nopat / invested_capital).replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame({
        "Revenue": revenue, "EPS": eps, "Equity": equity, "FCF": fcf, "ROIC": roic, "SharesDiluted": diluted_shares
    }).sort_index().tail(11)
    return df, years, "Alpha Vantage"

# -------------------- FMP fetch --------------------
FMP_BASE = "https://financialmodelingprep.com/api/v3"

def fmp_get(path: str, apikey: str, params=None):
    if params is None: params = {}
    params["apikey"] = apikey
    r = requests.get(f"{FMP_BASE}/{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fmp_series(reports, field) -> pd.Series:
    if not reports: return pd.Series(dtype="float64")
    rows = []
    for rep in reports:
        y = pd.to_datetime(rep.get("date") or rep.get("calendarYear"), errors="coerce").year
        if pd.isna(y):
            try: y = int(rep.get("calendarYear"))
            except: continue
        val = pd.to_numeric(rep.get(field, None), errors="coerce")
        rows.append((int(y), val))
    if not rows: return pd.Series(dtype="float64")
    s = pd.Series(dict(rows)).sort_index()
    return s.iloc[-11:].astype("float64")

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_fmp(symbol: str, apikey: str):
    inc = fmp_get(f"income-statement/{symbol}", apikey, {"period":"annual", "limit": 40})
    bal = fmp_get(f"balance-sheet-statement/{symbol}", apikey, {"period":"annual", "limit": 40})
    cfs = fmp_get(f"cash-flow-statement/{symbol}", apikey, {"period":"annual", "limit": 40})

    revenue        = fmp_series(inc, "revenue")
    net_income     = fmp_series(inc, "netIncome")
    diluted_eps    = fmp_series(inc, "epsdiluted")
    diluted_shares = fmp_series(inc, "weightedAverageShsOutDil")
    ebit           = fmp_series(inc, "ebit")
    tax_expense    = fmp_series(inc, "incomeTaxExpense")
    pretax_income  = fmp_series(inc, "incomeBeforeTax")

    equity     = fmp_series(bal, "totalStockholdersEquity")
    total_debt = fmp_series(bal, "totalDebt")
    cash       = fmp_series(bal, "cashAndCashEquivalents")

    cfo   = fmp_series(cfs, "netCashProvidedByOperatingActivities")
    capex = fmp_series(cfs, "capitalExpenditure")

    years = sorted(set(revenue.index) | set(net_income.index) | set(diluted_eps.index) |
                   set(diluted_shares.index) | set(ebit.index) | set(tax_expense.index) |
                   set(pretax_income.index) | set(equity.index) | set(total_debt.index) |
                   set(cash.index) | set(cfo.index) | set(capex.index))[-11:]

    def A(s): return s.reindex(years).astype("float64") if years else pd.Series(dtype="float64")
    revenue, net_income, diluted_eps, diluted_shares, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex = [
        A(x) for x in [revenue, net_income, diluted_eps, diluted_shares, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex]
    ]

    eps = diluted_eps.copy()
    if eps.isna().all() and not net_income.isna().all() and not diluted_shares.isna().all():
        with np.errstate(invalid="ignore", divide="ignore"):
            eps = net_income / diluted_shares.replace({0: np.nan})

    fcf = (cfo - capex) if (not cfo.isna().all() and not capex.isna().all()) else pd.Series([np.nan]*len(years), index=years)

    with np.errstate(divide="ignore", invalid="ignore"):
        tax_rate = (tax_expense / pretax_income).clip(0, 1)
    nopat = (ebit * (1 - tax_rate.fillna(0.21))) if not ebit.isna().all() else net_income
    invested_capital = (total_debt.fillna(0) + equity.fillna(0) - cash.fillna(0)).replace({0: np.nan})
    with np.errstate(divide="ignore", invalid="ignore"):
        roic = (nopat / invested_capital).replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame({
        "Revenue": revenue, "EPS": eps, "Equity": equity, "FCF": fcf, "ROIC": roic, "SharesDiluted": diluted_shares
    }).sort_index().tail(11)
    return df, years, "FMP"

# -------------------- Current Price --------------------
def get_price_alpha_vantage(symbol: str, apikey: str) -> float:
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={apikey}"
        j = requests.get(url, timeout=30).json()
        p = float(j.get("Global Quote", {}).get("05. price", "nan"))
        return p
    except Exception:
        return np.nan

def get_price_fmp(symbol: str, apikey: str) -> float:
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote-short/{symbol}?apikey={apikey}"
        j = requests.get(url, timeout=30).json()
        if isinstance(j, list) and j:
            return float(j[0].get("price", "nan"))
        return np.nan
    except Exception:
        return np.nan

# -------------------- Intrinsic Value Models --------------------
def intrinsic_eps_model(eps_last: float, growth: float, years: int, terminal_pe: float, discount: float) -> float:
    if pd.isna(eps_last) or eps_last <= 0: return np.nan
    eps_future = eps_last * ((1 + growth) ** years)
    future_price = eps_future * terminal_pe
    intrinsic_now = future_price / ((1 + discount) ** years)
    return intrinsic_now

def intrinsic_dcf_fcf(fcf_per_share_last: float, growth: float, years: int, terminal_g: float, discount: float) -> float:
    if pd.isna(fcf_per_share_last) or fcf_per_share_last <= 0: return np.nan
    if discount <= terminal_g: return np.nan  # avoid div-by-zero in terminal value

    pv = 0.0
    f = fcf_per_share_last
    for t in range(1, years + 1):
        f = f * (1 + growth)
        pv += f / ((1 + discount) ** t)
    # Terminal value (Gordon)
    f_next = f * (1 + terminal_g)
    tv = f_next / (discount - terminal_g)
    pv += tv / ((1 + discount) ** years)
    return pv

# -------------------- Run Search --------------------
if run:
    using_av = (provider == "Alpha Vantage")
    key_ok = (AV_KEY if using_av else FMP_KEY)
    st.info(f"Provider: **{provider}** · API key set: **{'Yes' if key_ok else 'No'}** · Ticker: **{ticker}**")

    if not key_ok:
        st.error("Missing API key in Streamlit Secrets.")
        df, years = demo_msft_df()
        source = "Demo (sample)"
        current_price = np.nan
    else:
        try:
            if using_av:
                df, years, source = fetch_alpha_vantage(ticker, AV_KEY)
                current_price = get_price_alpha_vantage(ticker, AV_KEY)
            else:
                df, years, source = fetch_fmp(ticker, FMP_KEY)
                current_price = get_price_fmp(ticker, FMP_KEY)
            if df.empty:
                st.warning("No data returned — switching to Demo Mode.")
                df, years = demo_msft_df()
                source = "Demo (sample)"
                current_price = np.nan
        except Exception as e:
            st.error(f"Error: {e}")
            df, years = demo_msft_df()
            source = "Demo (sample)"
            current_price = np.nan

    # -------------------- Big 5 Metrics --------------------
    n_years = max(len(df.index) - 1, 0)
    sales_cagr = cagr(df["Revenue"], n_years)
    eps_cagr   = cagr(df["EPS"], n_years)
    eqty_cagr  = cagr(df["Equity"], n_years)
    fcf_cagr   = cagr(df["FCF"], n_years)
    roic_avg   = safe_mean(df["ROIC"])

    def pf(v): return "PASS ✅" if not pd.isna(v) and v >= 0.10 else ("—" if pd.isna(v) else "FAIL ❌")
    def fmt(x): return "—" if pd.isna(x) else f"{x*100:.1f}%"

    # KPI cards
    colA, colB, colC, colD, colE = st.columns(5)
    colA.metric("Sales CAGR", fmt(sales_cagr))
    colB.metric("EPS CAGR",   fmt(eps_cagr))
    colC.metric("Equity CAGR",fmt(eqty_cagr))
    colD.metric("FCF CAGR",   fmt(fcf_cagr))
    colE.metric("Avg ROIC",   fmt(roic_avg))

    results = pd.DataFrame({
        "Metric": ["Sales (Revenue) CAGR","EPS CAGR","Equity CAGR","FCF CAGR","ROIC (10-yr Avg)"],
        "Value":  [fmt(sales_cagr), fmt(eps_cagr), fmt(eqty_cagr), fmt(fcf_cagr), fmt(roic_avg)],
        "Pass ≥10%?": [pf(sales_cagr), pf(eps_cagr), pf(eqty_cagr), pf(fcf_cagr), pf(roic_avg)]
    })

    st.subheader(f"Big 5 Results — {ticker}  ·  Source: {source}")
    st.dataframe(results, use_container_width=True)

    # ===== ROIC Breakdown =====
    st.markdown("#### ROIC Breakdown (from the 10-year window)")
    roic_series = df.get("ROIC", pd.Series(dtype=float)).astype(float).replace([np.inf, -np.inf], np.nan)
    roic_valid = roic_series.dropna()

    roic_10yr_avg = safe_mean(roic_series)
    roic_first5   = safe_mean(roic_valid.iloc[:5]) if len(roic_valid) >= 5 else np.nan
    roic_last3    = safe_mean(roic_valid.iloc[-3:]) if len(roic_valid) >= 1 else np.nan
    roic_last1    = roic_valid.iloc[-1] if len(roic_valid) >= 1 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROIC — 10-yr Avg", fmt(roic_10yr_avg))
    c2.metric("ROIC — First 5-yr Avg", fmt(roic_first5))
    c3.metric("ROIC — Last 3-yr Avg", fmt(roic_last3))
    c4.metric("ROIC — Last 1-yr", fmt(roic_last1))
    st.caption("ROIC = NOPAT / (Debt + Equity − Cash). First 5-yr uses the oldest five years in the displayed 10-year window.")

    # ===== Data Coverage =====
    st.markdown("#### Data Coverage (non-missing values used)")
    coverage = df.notna().sum().rename("Valid Years").to_frame()
    coverage["Out of"] = len(df.index)
    st.dataframe(coverage.T, use_container_width=True)

    # ===== Intrinsic Value =====
    st.markdown("### Intrinsic Value")
    last_eps = df["EPS"].dropna().iloc[-1] if df["EPS"].notna().any() else np.nan

    # FCF per share (uses last year FCF and SharesDiluted)
    shares_last = df["SharesDiluted"].dropna().iloc[-1] if "SharesDiluted" in df and df["SharesDiluted"].notna().any() else np.nan
    fcf_last = df["FCF"].dropna().iloc[-1] if df["FCF"].notna().any() else np.nan
    fcf_per_share_last = (fcf_last / shares_last) if (not pd.isna(fcf_last) and not pd.isna(shares_last) and shares_last > 0) else np.nan

    iv_eps = intrinsic_eps_model(last_eps, growth_eps, years_eps, terminal_pe, discount)
    iv_dcf = intrinsic_dcf_fcf(fcf_per_share_last, growth_fcf, years_dcf, terminal_g, discount)

    colv1, colv2, colv3 = st.columns(3)
    colv1.metric("Intrinsic (EPS / P-E)", f"${iv_eps:,.2f}" if not pd.isna(iv_eps) else "—")
    colv2.metric("Intrinsic (FCF DCF / sh.)", f"${iv_dcf:,.2f}" if not pd.isna(iv_dcf) else "—")
    colv3.metric("Current Price", f"${current_price:,.2f}" if not pd.isna(current_price) else "—")

    if not pd.isna(current_price):
        if not pd.isna(iv_eps):
            mos_eps = (iv_eps - current_price) / current_price
            st.write(f"**Margin of Safety (EPS model):** {pct(mos_eps)}")
        if not pd.isna(iv_dcf):
            mos_dcf = (iv_dcf - current_price) / current_price
            st.write(f"**Margin of Safety (FCF DCF):** {pct(mos_dcf)}")

    with st.expander("Valuation Inputs Recap"):
        rec = {
            "EPS last (ttm/annual)": last_eps,
            "FCF last": fcf_last,
            "Shares diluted last": shares_last,
            "FCF per share last": fcf_per_share_last,
            "EPS growth": growth_eps, "Years (EPS)": years_eps, "Terminal P/E": terminal_pe,
            "FCF growth": growth_fcf, "Years (DCF)": years_dcf, "Terminal g": terminal_g,
            "Discount rate": discount
        }
        st.json({k: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.floating)) else v)) for k, v in rec.items()})

    # ===== Value-Investor Summary (OpenAI) =====
    st.markdown("### Value-Investor Summary (OpenAI)")
    if not OPENAI_KEY:
        st.info("Add **OPENAI_API_KEY** in Secrets to enable the summary.")
    else:
        want = st.button("Generate Summary with OpenAI")
        if want:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_KEY)
                # Build a compact context for the LLM
                context = {
                    "ticker": ticker,
                    "years": list(map(int, df.index.tolist())),
                    "metrics": {
                        "sales_cagr": float(sales_cagr) if not pd.isna(sales_cagr) else None,
                        "eps_cagr": float(eps_cagr) if not pd.isna(eps_cagr) else None,
                        "equity_cagr": float(eqty_cagr) if not pd.isna(eqty_cagr) else None,
                        "fcf_cagr": float(fcf_cagr) if not pd.isna(fcf_cagr) else None,
                        "roic_avg": float(roic_avg) if not pd.isna(roic_avg) else None,
                        "roic_first5": float(roic_first5) if not pd.isna(roic_first5) else None,
                        "roic_last3": float(roic_last3) if not pd.isna(roic_last3) else None,
                        "roic_last1": float(roic_last1) if not pd.isna(roic_last1) else None,
                    },
                    "intrinsic": {
                        "iv_eps": float(iv_eps) if not pd.isna(iv_eps) else None,
                        "iv_dcf": float(iv_dcf) if not pd.isna(iv_dcf) else None,
                        "current_price": float(current_price) if not pd.isna(current_price) else None
                    }
                }
                prompt = (
                    "You are a disciplined value investor (Phil Town style). "
                    "Summarize the company strictly from a value perspective using the data below: "
                    f"{context}. "
                    "Explain if it passes the 10% rule on each metric, comment on ROIC trend "
                    "(first 5 vs last 3 vs last 1), and interpret intrinsic values versus current price. "
                    "Keep it under 180 words, concise, neutral tone."
                )
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content": prompt}],
                    temperature=0.3,
                    max_tokens=350,
                )
                st.write(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"OpenAI error: {e}")

    with st.expander("Raw series used"):
        st.dataframe(df)

    with st.expander("Mini charts"):
        c1, c2, c3 = st.columns(3)
        c1.line_chart(df[["Revenue","FCF"]].dropna(), height=220)
        c2.line_chart(df[["EPS"]].dropna(), height=220)
        c3.line_chart(df[["ROIC"]].dropna(), height=220)

else:
    st.info("Enter a ticker and click **Search**. Choose provider in the sidebar and confirm your API key in Secrets.")
