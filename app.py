import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

st.set_page_config(page_title="Phil Town Big 5 Screener", layout="wide")
st.title("Phil Town Big 5 — 10-Year Screener")
st.caption("Alpha Vantage fundamentals → 10-year CAGRs for Sales, EPS, Equity, FCF, plus 10-yr Avg ROIC. Pass rule: ≥ 10%.")

API_KEY = st.secrets.get("ALPHAVANTAGE_API_KEY", "").strip()
BASE = "https://www.alphavantage.co/query"

def av_get(fn, symbol):
    """Call Alpha Vantage endpoint and return annual reports list (newest-first)."""
    params = {"function": fn, "symbol": symbol, "apikey": API_KEY}
    r = requests.get(BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    reports = data.get("annualReports") or []
    return reports

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_alpha_vantage(symbol: str):
    # free tier: 5 req/min → tiny spacing across 3 calls
    inc = av_get("INCOME_STATEMENT", symbol); time.sleep(0.6)
    bal = av_get("BALANCE_SHEET", symbol);   time.sleep(0.6)
    cfs = av_get("CASH_FLOW", symbol);       time.sleep(0.6)

    def series(reports, field):
        if not reports: return pd.Series(dtype="float64")
        rows = []
        for rep in reports:
            y = pd.to_datetime(rep.get("fiscalDateEnding", ""), errors="coerce").year
            if pd.isna(y):
                continue
            val = pd.to_numeric(rep.get(field, None), errors="coerce")
            rows.append((int(y), val))
        if not rows: return pd.Series(dtype="float64")
        s = pd.Series(dict(rows)).sort_index()
        return s.iloc[-11:].astype("float64")  # keep most recent ~10 intervals

    # Income Statement
    revenue        = series(inc, "totalRevenue")
    net_income     = series(inc, "netIncome")
    diluted_eps    = series(inc, "dilutedEPS")
    diluted_shares = series(inc, "weightedAverageShsOutDil")
    ebit           = series(inc, "ebit")
    tax_expense    = series(inc, "incomeTaxExpense")
    pretax_income  = series(inc, "incomeBeforeTax")

    # Balance Sheet
    equity         = series(bal, "totalShareholderEquity")
    total_debt     = series(bal, "totalDebt")
    if total_debt.empty:
        total_debt = series(bal, "shortLongTermDebtTotal")
    cash           = series(bal, "cashAndCashEquivalentsAtCarryingValue")
    if cash.empty:
        cash = series(bal, "cashAndShortTermInvestments")

    # Cash Flow
    cfo            = series(cfs, "operatingCashflow")
    capex          = series(cfs, "capitalExpenditures")

    # Align years
    years = sorted(
        set(revenue.index) | set(net_income.index) | set(diluted_eps.index) |
        set(diluted_shares.index) | set(ebit.index) | set(tax_expense.index) |
        set(pretax_income.index) | set(equity.index) | set(total_debt.index) |
        set(cash.index) | set(cfo.index) | set(capex.index)
    )[-11:]

    def align(s): return s.reindex(years).astype("float64") if years else pd.Series(dtype="float64")

    revenue, net_income, diluted_eps, diluted_shares, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex = [
        align(x) for x in [revenue, net_income, diluted_eps, diluted_shares, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex]
    ]

    # EPS: prefer reported diluted EPS; else NI / diluted shares (Phil Town style)
    eps = diluted_eps.copy()
    if eps.isna().all() and not net_income.isna().all() and not diluted_shares.isna().all():
        with np.errstate(invalid="ignore", divide="ignore"):
            eps = net_income / diluted_shares.replace({0: np.nan})

    # FCF = CFO − CapEx
    fcf = (cfo - capex) if (not cfo.isna().all() and not capex.isna().all()) else pd.Series([np.nan]*len(years), index=years)

    # Tax rate (cap at [0,1] to avoid weirdness)
    with np.errstate(divide="ignore", invalid="ignore"):
        tax_rate = (tax_expense / pretax_income).clip(0, 1)

    # ROIC proxy = NOPAT / (Debt + Equity − Cash). Fallback to NI if EBIT missing.
    nopat = (ebit * (1 - tax_rate.fillna(0.21))) if not ebit.isna().all() else net_income
    invested_capital = (total_debt.fillna(0) + equity.fillna(0) - cash.fillna(0)).replace({0: np.nan})
    with np.errstate(divide="ignore", invalid="ignore"):
        roic = (nopat / invested_capital).replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame({
        "Revenue": revenue, "EPS": eps, "Equity": equity, "FCF": fcf, "ROIC": roic
    }).sort_index().tail(11)

    return df, years

def CAGR(series: pd.Series, years_count: int):
    try:
        if years_count <= 0: return np.nan
        first, last = series.iloc[0], series.iloc[-1]
        if pd.isna(first) or pd.isna(last) or first <= 0 or last <= 0: return np.nan
        return (last / first) ** (1 / years_count) - 1
    except Exception:
        return np.nan

def pct(x): return "—" if (x is None or pd.isna(x)) else f"{x*100:.1f}%"

ticker = st.text_input("Enter ticker (e.g., AAPL, MSFT, ADBE):", value="AAPL").strip().upper()

if not API_KEY:
    st.error("Missing API key. In Streamlit: ⋯ → Manage app → Settings → Secrets\n\nALPHAVANTAGE_API_KEY = your_key_here")
else:
    try:
        df, years = fetch_alpha_vantage(ticker)
        if df.empty:
            st.warning("No data returned. Try a large-cap US ticker (AAPL/MSFT) or wait 60s (free API rate limit).")
        n_years = max(len(years) - 1, 0)

        col1, col2 = st.columns([1,2])
        with col1:
            st.subheader("Years used")
            st.write(years if years else "—")
            st.caption(f"Span: {n_years} year(s) of growth. Source: Alpha Vantage")

        # Big 5
        sales_cagr = CAGR(df["Revenue"].dropna(), n_years) if "Revenue" in df else np.nan
        eps_cagr   = CAGR(df["EPS"].dropna(),     n_years) if "EPS"     in df else np.nan
        eqty_cagr  = CAGR(df["Equity"].dropna(),  n_years) if "Equity"  in df else np.nan
        fcf_cagr   = CAGR(df["FCF"].dropna(),     n_years) if "FCF"     in df else np.nan
        roic_avg   = df["ROIC"].replace([np.inf, -np.inf], np.nan).dropna().mean() if "ROIC" in df else np.nan

        def passfail(v): return "—" if pd.isna(v) else ("PASS ✅" if v >= 0.10 else "FAIL ❌")

        results = pd.DataFrame({
            "Metric": ["Sales (Revenue) CAGR","EPS CAGR","Equity CAGR","FCF CAGR","ROIC (10-yr Avg)"],
            "Value":  [pct(sales_cagr), pct(eps_cagr), pct(eqty_cagr), pct(fcf_cagr), pct(roic_avg)],
            "Pass ≥10%?": [passfail(sales_cagr), passfail(eps_cagr), passfail(eqty_cagr), passfail(fcf_cagr), passfail(roic_avg)]
        })

        with col2:
            st.subheader(f"Big 5 Results — {ticker}")
            st.dataframe(results, use_container_width=True)

        with st.expander("Raw data used"):
            st.dataframe(df)

        st.caption("Notes: EPS prefers diluted EPS; else NI ÷ diluted shares. FCF = CFO − CapEx. ROIC proxy = NOPAT / (Debt + Equity − Cash).")
    except Exception as e:
        st.exception(e)
