import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Phil Town Big 5 Screener", layout="wide")
st.title("Phil Town Big 5 — 10-Year Screener")
st.caption("Fetches ~10 years of fundamentals via Yahoo Finance and checks Sales, EPS, Equity, FCF CAGRs ≥ 10%, and 10-yr Avg ROIC ≥ 10%.")

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_statements(ticker):
    tk = yf.Ticker(ticker)

    inc = bal = cfs = pd.DataFrame()

    # Try modern yfinance API (newer versions)
    try:
        inc = tk.get_income_stmt(freq="annual")
        bal = tk.get_balance_sheet(freq="annual")
        cfs = tk.get_cashflow(freq="annual")
    except Exception:
        pass

    # Fallback to legacy attributes (older Streamlit images sometimes have these)
    if inc is None or inc.empty:
        try: inc = tk.financials
        except Exception: inc = pd.DataFrame()
    if bal is None or bal.empty:
        try: bal = tk.balance_sheet
        except Exception: bal = pd.DataFrame()
    if cfs is None or cfs.empty:
        try: cfs = tk.cashflow
        except Exception: cfs = pd.DataFrame()

    def tidy(df):
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        # Safely coerce date-like columns then keep the most recent 11 points (~10 intervals)
        df.columns = pd.to_datetime(df.columns, errors="coerce").year
        years = [y for y in sorted(df.columns) if pd.notna(y)]
        years = years[-11:]
        return df[years] if years else pd.DataFrame()

    return tidy(inc), tidy(bal), tidy(cfs)

def pick(df, names):
    if df is None or df.empty: return pd.Series(dtype="float64")
    lower = {str(i).strip().lower(): i for i in df.index}
    for n in names:
        key = str(n).strip().lower()
        if key in lower: return df.loc[lower[key]]
    # fuzzy contains fallback
    for idx in df.index:
        s = str(idx).lower().replace(" ", "")
        for n in names:
            if str(n).lower().replace(" ", "") in s:
                return df.loc[idx]
    return pd.Series(dtype="float64")

def compute_series(ticker):
    inc, bal, cfs = fetch_statements(ticker)

    revenue        = pick(inc, ["Total Revenue","Revenue"])
    net_income     = pick(inc, ["Net Income","NetIncome"])
    diluted_shares = pick(inc, ["Diluted Average Shares","Weighted Average Diluted Shares Outstanding","Diluted Shares"])
    diluted_eps    = pick(inc, ["Diluted EPS","DilutedEPS"])
    ebit           = pick(inc, ["EBIT","Operating Income","Earnings Before Interest and Taxes"])

    equity     = pick(bal, ["Total Stockholder Equity","TotalStockholderEquity","Shareholders' Equity","Total Equity"])
    total_debt = pick(bal, ["Total Debt","Short Long Term Debt","ShortLongTermDebtTotal"])
    cash       = pick(bal, ["Cash And Cash Equivalents","Cash And Cash Equivalents At Carrying Value","Cash","CashAndCashEquivalents"])

    cfo   = pick(cfs, ["Operating Cash Flow","Total Cash From Operating Activities","NetCashProvidedByUsedInOperatingActivities"])
    capex = pick(cfs, ["Capital Expenditure","CapitalExpenditures","Investments In Property Plant And Equipment"])

    years = sorted(set(revenue.index) | set(net_income.index) | set(diluted_shares.index) |
                   set(diluted_eps.index) | set(ebit.index) | set(equity.index) |
                   set(total_debt.index) | set(cash.index) | set(cfo.index) | set(capex.index))[-11:]

    def align(s): return s.reindex(years).astype("float64") if len(years) else pd.Series(dtype="float64")

    revenue, net_income, diluted_shares, diluted_eps, ebit, equity, total_debt, cash, cfo, capex = [
        align(x) for x in [revenue, net_income, diluted_shares, diluted_eps, ebit, equity, total_debt, cash, cfo, capex]
    ]

    # EPS: prefer reported diluted EPS; else NI / diluted shares
    eps = diluted_eps.copy()
    if eps.isna().all() and not net_income.isna().all() and not diluted_shares.isna().all():
        with np.errstate(invalid="ignore", divide="ignore"):
            eps = net_income / diluted_shares.replace({0: np.nan})

    # FCF = CFO − CapEx (capex may be negative; still do CFO - CapEx)
    fcf = (cfo - capex) if (not cfo.isna().all() and not capex.isna().all()) else pd.Series([np.nan]*len(years), index=years)

    # ROIC proxy: NOPAT / (Debt + Equity − Cash)
    tax_expense   = pick(inc, ["Income Tax Expense","IncomeTaxExpense"]).reindex(years)
    pretax_income = pick(inc, ["Income Before Tax","IncomeBeforeTax","Earnings Before Tax"]).reindex(years)
    with np.errstate(divide="ignore", invalid="ignore"):
        tax_rate = (tax_expense / pretax_income).clip(0,1)

    if ebit.isna().all():
        ebit = pick(inc, ["Operating Income","OperatingIncome"]).reindex(years)

    invested_capital = (total_debt.fillna(0) + equity.fillna(0) - cash.fillna(0)).replace({0: np.nan})
    nopat = (ebit * (1 - tax_rate.fillna(0.21))) if not ebit.isna().all() else net_income
    with np.errstate(divide="ignore", invalid="ignore"):
        roic_series = (nopat / invested_capital).replace([np.inf, -np.inf], np.nan)

    return pd.DataFrame({"Revenue": revenue, "EPS": eps, "Equity": equity, "FCF": fcf, "ROIC": roic_series})

def CAGR(series, years_count):
    try:
        first, last = series.iloc[0], series.iloc[-1]
        if pd.isna(first) or pd.isna(last) or first <= 0 or last <= 0 or years_count <= 0:
            return np.nan
        return (last/first)**(1/years_count) - 1
    except Exception:
        return np.nan

def pct(x): return None if pd.isna(x) else f"{x*100:.1f}%"

ticker = st.text_input("Enter ticker (e.g., AAPL, MSFT, ADBE):", value="AAPL").strip().upper()

if st.button("Run Screener") or ticker:
    try:
        st.info("Fetching data…")
        df = compute_series(ticker).sort_index().tail(11)
        if df.empty:
            st.warning("No data pulled. Try another ticker or refresh. If this persists, yfinance may be rate-limiting.")
        n_years = max(len(df.index) - 1, 0)

        col1, col2 = st.columns([1,2])
        with col1:
            st.subheader("Years used")
            st.write(list(df.index))
            st.caption(f"Span: {n_years} year(s) of growth.")

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

        checks  = [sales_cagr, eps_cagr, eqty_cagr, fcf_cagr, roic_avg]
        valid   = [v for v in checks if not pd.isna(v)]
        overall = (len([v for v in valid if v >= 0.10]) / 5) if valid else np.nan
        st.metric("Overall Big-5 Pass Rate", pct(overall) if not pd.isna(overall) else "—")

        with st.expander("Show raw data"):
            st.dataframe(df, use_container_width=True)

        st.caption("Notes: EPS prefers diluted EPS; otherwise NI ÷ Diluted Shares. FCF = CFO − CapEx. ROIC proxy = NOPAT / (Debt + Equity − Cash).")
    except Exception as e:
        st.exception(e)
        st.stop()
