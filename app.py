import streamlit as st
import pandas as pd
import numpy as np
import requests

# -------------------- Page Config --------------------
st.set_page_config(page_title="Phil Town Big 5 Screener", layout="wide")
st.title("Phil Town Big 5 — 10-Year Screener")
st.caption("Checks Sales, EPS, Equity, FCF CAGR ≥ 10%, and 10-yr Avg ROIC ≥ 10%.")

# -------------------- Cache Control --------------------
scol1, scol2 = st.columns([3, 1])
with scol1:
    st.info("If results don't show, try **Clear Cache** then search again.")
with scol2:
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared. Run a new search.")

# -------------------- Sidebar --------------------
st.sidebar.header("Settings")
provider = st.sidebar.radio("Data Provider", ["Alpha Vantage", "FMP"])
ticker = st.text_input("Enter ticker (e.g., AAPL, MSFT, ADBE):", value="AAPL").strip().upper()
run = st.button("Search")

AV_KEY = st.secrets.get("ALPHAVANTAGE_API_KEY")
FMP_KEY = st.secrets.get("FMP_API_KEY")

# -------------------- Demo Mode --------------------
def demo_msft_df():
    years = list(range(2015, 2025))
    df = pd.DataFrame({
        "Revenue": [93580, 85320, 89950, 110360, 125843, 143015, 168088, 198270, 211915, 245000],
        "EPS":     [2.48, 2.79, 3.31, 2.13, 5.76, 6.20, 8.05, 9.21, 9.68, 11.00],
        "Equity":  [72163, 82718, 82572, 82572, 118304, 118304, 166542, 166542, 194000, 210000],
        "FCF":     [23969, 31378, 31922, 32694, 45230, 45300, 56300, 65700, 67800, 78000],
        "ROIC":    [0.12, 0.13, 0.15, 0.10, 0.18, 0.19, 0.21, 0.22, 0.20, 0.21],
    }, index=years).astype(float)
    return df.iloc[-10:], years[-10:]

# -------------------- Fetchers --------------------
def fetch_alpha_vantage(ticker, key):
    url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={key}"
    inc = requests.get(url).json()
    url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={key}"
    bal = requests.get(url).json()
    url = f"https://www.alphavantage.co/query?function=CASH_FLOW&symbol={ticker}&apikey={key}"
    cfs = requests.get(url).json()

    # Parse minimal Big 5 data
    def extract(data, field):
        try:
            return {int(x['fiscalDateEnding'][:4]): float(x.get(field, None)) for x in data.get('annualReports', [])}
        except:
            return {}
    revenue = extract(inc, "totalRevenue")
    eps = extract(inc, "dilutedEps")
    equity = extract(bal, "totalShareholderEquity")
    cfo = extract(cfs, "operatingCashflow")
    capex = extract(cfs, "capitalExpenditures")
    fcf = {y: (cfo.get(y, 0) - abs(capex.get(y, 0))) for y in cfo}

    # Fake ROIC for now (needs more calc)
    roic = {y: np.nan for y in revenue}

    years = sorted(revenue.keys())[-10:]
    df = pd.DataFrame({
        "Revenue": [revenue.get(y, np.nan) for y in years],
        "EPS": [eps.get(y, np.nan) for y in years],
        "Equity": [equity.get(y, np.nan) for y in years],
        "FCF": [fcf.get(y, np.nan) for y in years],
        "ROIC": [roic.get(y, np.nan) for y in years],
    }, index=years)
    return df, years, "Alpha Vantage"

def fetch_fmp(ticker, key):
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=10&apikey={key}"
    inc = requests.get(url).json()
    url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?limit=10&apikey={key}"
    bal = requests.get(url).json()
    url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?limit=10&apikey={key}"
    cfs = requests.get(url).json()

    def extract(data, field):
        try:
            return {int(x['date'][:4]): float(x.get(field, None)) for x in data}
        except:
            return {}
    revenue = extract(inc, "revenue")
    eps = extract(inc, "eps")
    equity = extract(bal, "totalStockholdersEquity")
    cfo = extract(cfs, "netCashProvidedByOperatingActivities")
    capex = extract(cfs, "capitalExpenditure")
    fcf = {y: (cfo.get(y, 0) - abs(capex.get(y, 0))) for y in cfo}

    # Fake ROIC for now
    roic = {y: np.nan for y in revenue}

    years = sorted(revenue.keys())[-10:]
    df = pd.DataFrame({
        "Revenue": [revenue.get(y, np.nan) for y in years],
        "EPS": [eps.get(y, np.nan) for y in years],
        "Equity": [equity.get(y, np.nan) for y in years],
        "FCF": [fcf.get(y, np.nan) for y in years],
        "ROIC": [roic.get(y, np.nan) for y in years],
    }, index=years)
    return df, years, "FMP"

# -------------------- Run Search --------------------
if run:
    using_av = (provider == "Alpha Vantage")
    key_ok = (AV_KEY if using_av else FMP_KEY)
    st.info(f"Provider: **{provider}** · API key set: **{'Yes' if key_ok else 'No'}**")

    if not key_ok:
        st.error("Missing API key in Streamlit Secrets.")
        df, years = demo_msft_df()
        source = "Demo (sample)"
    else:
        try:
            if using_av:
                df, years, source = fetch_alpha_vantage(ticker, AV_KEY)
            else:
                df, years, source = fetch_fmp(ticker, FMP_KEY)
            if df.empty:
                st.warning("No data returned — switching to Demo Mode.")
                df, years = demo_msft_df()
                source = "Demo (sample)"
        except Exception as e:
            st.error(f"Error: {e}")
            df, years = demo_msft_df()
            source = "Demo (sample)"

    # -------------------- Big 5 Metrics --------------------
    n_years = max(len(years) - 1, 0)
    def cagr(series, years):
        try:
            first, last = series.iloc[0], series.iloc[-1]
            if pd.isna(first) or pd.isna(last) or first <= 0 or last <= 0 or years <= 0:
                return np.nan
            return (last/first)**(1/years) - 1
        except:
            return np.nan
    def pct(x): return "—" if pd.isna(x) else f"{x*100:.1f}%"
    def pf(v): return "PASS ✅" if not pd.isna(v) and v >= 0.10 else ("—" if pd.isna(v) else "FAIL ❌")

    sales_cagr = cagr(df["Revenue"], n_years)
    eps_cagr   = cagr(df["EPS"], n_years)
    eqty_cagr  = cagr(df["Equity"], n_years)
    fcf_cagr   = cagr(df["FCF"], n_years)
    roic_avg   = df["ROIC"].mean()

    results = pd.DataFrame({
        "Metric": ["Sales (Revenue) CAGR","EPS CAGR","Equity CAGR","FCF CAGR","ROIC (10-yr Avg)"],
        "Value":  [pct(sales_cagr), pct(eps_cagr), pct(eqty_cagr), pct(fcf_cagr), pct(roic_avg)],
        "Pass ≥10%?": [pf(sales_cagr), pf(eps_cagr), pf(eqty_cagr), pf(fcf_cagr), pf(roic_avg)]
    })

    st.subheader(f"Big 5 Results — {ticker}  ·  Source: {source}")
    st.dataframe(results, use_container_width=True)

    with st.expander("Raw data used"):
        st.dataframe(df)

else:
    st.info("Enter a ticker and click **Search**. Use the sidebar to choose provider & confirm your API key in Secrets.")
