import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Phil Town Big 5 Screener", layout="wide")
st.title("Phil Town Big 5 — 10-Year Screener")
st.caption("Auto-fetches ~10 years of fundamentals via Yahoo Finance and checks Sales, EPS, Equity, FCF CAGRs ≥ 10%, and 10-yr Avg ROIC ≥ 10%.")

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_statements(ticker):
    tk = yf.Ticker(ticker)
    inc = bal = cfs = pd.DataFrame()

    # Try modern API
    try:
        inc = tk.get_income_stmt(freq="annual")
        bal = tk.get_balance_sheet(freq="annual")
        cfs = tk.get_cashflow(freq="annual")
    except Exception:
        pass

    # Fallback to legacy
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
        # Columns are dates → convert to year
        df.columns = pd.to_datetime(df.columns, errors="coerce").year
        years = [y for y in sorted(df.columns) if pd.notna(y)]
        years = years[-11:]
        df = df[years] if years else pd.DataFrame()
        # Force numeric
        df = df.apply(pd.to_numeric, errors="coerce")
        return df

    return tidy(inc), tidy(bal), tidy(cfs)

def pick(df, names):
    """Find a row by any of the provided names (case/space insensitive, also substring fallback)."""
    if df is None or df.empty: 
        return pd.Series(dtype="float64"), None
    idx_map = {str(i).strip().lower(): i for i in df.index}
    # Exact insensitive match
    for n in names:
        key = str(n).strip().lower()
        if key in idx_map:
            return df.loc[idx_map[key]], idx_map[key]
    # Substring fallback
    for idx in df.index:
        s = str(idx).lower().replace(" ", "")
        for n in names:
            if str(n).lower().replace(" ", "") in s:
                return df.loc[idx], idx
    return pd.Series(dtype="float64"), None

def compute_series(ticker):
    inc, bal, cfs = fetch_statements(ticker)

    # Broader label sets to cover banks/insurers and alt spellings
    revenue_names = [
        "Total Revenue","Revenue","Total Operating Revenues","Operating Revenue",
        "Revenues","Total Net Revenue"
    ]
    net_income_names = [
        "Net Income","NetIncome","Net Income Applicable To Common Shares","Net Income Common Stockholders"
    ]
    diluted_shares_names = [
        "Diluted Average Shares","Weighted Average Diluted Shares Outstanding","Diluted Shares",
        "Diluted Average Shares Outstanding","Weighted Average Shares Diluted"
    ]
    diluted_eps_names = ["Diluted EPS","DilutedEPS","EPS Diluted"]
    ebit_names = ["EBIT","Operating Income","Earnings Before Interest and Taxes","Operating Profit"]

    equity_names = [
        "Total Stockholder Equity","Total Stockholders' Equity","TotalStockholderEquity","Shareholders' Equity",
        "Total Equity","Total Shareholder Equity","Common Stock Equity","Total Common Equity"
    ]
    total_debt_names = [
        "Total Debt","Short Long Term Debt","ShortLongTermDebtTotal","Total Liabilities & Debt","Total Interest-Bearing Debt"
    ]
    cash_names = [
        "Cash And Cash Equivalents","Cash And Cash Equivalents At Carrying Value","Cash","CashAndCashEquivalents"
    ]

    cfo_names = [
        "Operating Cash Flow","Total Cash From Operating Activities","NetCashProvidedByUsedInOperatingActivities"
    ]
    capex_names = [
        "Capital Expenditure","CapitalExpenditures","Investments In Property Plant And Equipment","Purchase Of Property Plant And Equipment"
    ]
    tax_expense_names = ["Income Tax Expense","IncomeTaxExpense","Provision For Income Taxes"]
    pretax_income_names = ["Income Before Tax","IncomeBeforeTax","Earnings Before Tax","Pretax Income"]

    # Pick rows
    revenue, rev_row = pick(inc, revenue_names)
    net_income, ni_row = pick(inc, net_income_names)
    diluted_shares, sh_row = pick(inc, diluted_shares_names)
    diluted_eps, eps_row = pick(inc, diluted_eps_names)
    ebit, ebit_row = pick(inc, ebit_names)

    equity, eq_row = pick(bal, equity_names)
    total_debt, debt_row = pick(bal, total_debt_names)
    cash, cash_row = pick(bal, cash_names)

    cfo, cfo_row = pick(cfs, cfo_names)
    capex, capex_row = pick(cfs, capex_names)

    years = sorted(set(revenue.index) | set(net_income.index) | set(diluted_shares.index) |
                   set(diluted_eps.index) | set(ebit.index) | set(equity.index) |
                   set(total_debt.index) | set(cash.index) | set(cfo.index) | set(capex.index))[-11:]

    def align(s): 
        return s.reindex(years).astype("float64") if len(years) else pd.Series(dtype="float64")

    revenue, net_income, diluted_shares, diluted_eps, ebit, equity, total_debt, cash, cfo, capex = [
        align(x) for x in [revenue, net_income, diluted_shares, diluted_eps, ebit, equity, total_debt, cash, cfo, capex]
    ]

    # Build EPS if needed
    eps = diluted_eps.copy()
    if eps.isna().all() and not net_income.isna().all() and not diluted_shares.isna().all():
        with np.errstate(invalid="ignore", divide="ignore"):
            eps = net_income / diluted_shares.replace({0: np.nan})

    # FCF
    fcf = (cfo - capex) if (not cfo.isna().all() and not capex.isna().all()) else pd.Series([np.nan]*len(years), index=years)

    # ROIC proxy
    tax_expense, _ = pick(inc, tax_expense_names)
    pretax_income, _ = pick(inc, pretax_income_names)
    tax_expense, pretax_income = align(tax_expense), align(pretax_income)
    with np.errstate(divide="ignore", invalid="ignore"):
        tax_rate = (tax_expense / pretax_income).clip(0,1)

    if ebit.isna().all():
        ebit, _ = pick(inc, ["Operating Income","OperatingIncome"])
        ebit = align(ebit)

    invested_capital = (total_debt.fillna(0) + equity.fillna(0) - cash.fillna(0)).replace({0: np.nan})
    nopat = (ebit * (1 - tax_rate.fillna(0.21))) if not ebit.isna().all() else net_income
    with np.errstate(divide="ignore", invalid="ignore"):
        roic_series = (nopat / invested_capital).replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame({"Revenue": revenue, "EPS": eps, "Equity": equity, "FCF": fcf, "ROIC": roic_series})

    # For debugging UI
    found = {
        "Revenue row": rev_row, "Net income row": ni_row, "Diluted shares row": sh_row, "Diluted EPS row": eps_row, "EBIT row": ebit_row,
        "Equity row": eq_row, "Total debt row": debt_row, "Cash row": cash_row, "CFO row": cfo_row, "CapEx row": capex_row
    }

    return df, years, inc, bal, cfs, found

def CAGR(series, years_count):
    try:
        first, last = series.iloc[0], series.iloc[-1]
        if pd.isna(first) or pd.isna(last) or first <= 0 or last <= 0 or years_count <= 0:
            return np.nan
        return (last/first)**(1/years_count) - 1
    except Exception:
        return np.nan

def pct(x): 
    return "—" if (x is None or pd.isna(x)) else f"{x*100:.1f}%"

ticker = st.text_input("Enter ticker (e.g., AAPL, MSFT, ADBE):", value="AAPL").strip().upper()

# Auto-run on text input (no button)
try:
    df, years, inc, bal, cfs, found = compute_series(ticker)
    n_years = max(len(years) - 1, 0)

    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("Years used")
        st.write(years if years else "—")
        st.caption(f"Span: {n_years} year(s) of growth.")

    # Compute metrics
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

    with st.expander("What data did we actually pull? (raw statements)"):
        st.write("Income Statement (annual):")
        st.dataframe(inc)
        st.write("Balance Sheet (annual):")
        st.dataframe(bal)
        st.write("Cash Flow (annual):")
        st.dataframe(cfs)

    with st.expander("Which rows were found?"):
        st.json(found)

    with st.expander("Final aligned series (used for math)"):
        st.dataframe(df)

    st.caption("If a series is empty, check which row names were found above. Different industries (esp. banks/insurers) use different labels.")
except Exception as e:
    st.exception(e)
    st.stop()
