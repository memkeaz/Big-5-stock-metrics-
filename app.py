# ---------- Alpha Vantage (patched + clearer errors) ----------
AV_BASE  = "https://www.alphavantage.co/query"

def av_get(fn: str, symbol: str, apikey: str):
    params = {"function": fn, "symbol": symbol, "apikey": apikey}
    r = requests.get(AV_BASE, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()

    # Clear error messages from AV
    if isinstance(j, dict):
        if "Note" in j and j["Note"]:
            raise RuntimeError("Alpha Vantage rate limit hit. Wait a minute or switch provider (Yahoo/FMP).")
        if "Information" in j and j["Information"]:
            raise RuntimeError(f"Alpha Vantage error: {j['Information']}")
        if "Error Message" in j and j["Error Message"]:
            raise RuntimeError(f"Alpha Vantage error: {j['Error Message']}")

    # Expected shape for fundamentals
    annual = j.get("annualReports", [])
    quarterly = j.get("quarterlyReports", [])
    # Return both so the caller can also report diagnostics
    return annual, quarterly

def av_series(reports, field) -> pd.Series:
    if not reports: return pd.Series(dtype="float64")
    rows = []
    for rep in reports:
        y = pd.to_datetime(rep.get("fiscalDateEnding", ""), errors="coerce").year
        if pd.isna(y): continue
        rows.append((int(y), pd.to_numeric(rep.get(field), errors="coerce")))
    if not rows: return pd.Series(dtype="float64")
    s = pd.Series(dict(rows)).sort_index()
    return s.iloc[-11:].astype("float64")

def av_series_sum(reports, fields) -> pd.Series:
    total = None
    for f in fields:
        s = av_series(reports, f)
        total = s if total is None else total.add(s, fill_value=0)
    return total if total is not None else pd.Series(dtype="float64")

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_alpha_vantage(symbol: str, apikey: str):
    inc_a, inc_q = av_get("INCOME_STATEMENT", symbol, apikey)
    bal_a, bal_q = av_get("BALANCE_SHEET",  symbol, apikey)
    cfs_a, cfs_q = av_get("CASH_FLOW",      symbol, apikey)

    # If literally nothing came back, surface a clear message
    if not inc_a and not bal_a and not cfs_a:
        raise RuntimeError("Alpha Vantage returned no annual data. Try Yahoo or FMP, or wait due to rate limits.")

    # Income statement
    revenue        = av_series(inc_a, "totalRevenue")
    net_income     = av_series(inc_a, "netIncome")
    diluted_eps    = av_series(inc_a, "dilutedEPS")
    ebit           = av_series(inc_a, "ebit")
    tax_expense    = av_series(inc_a, "incomeTaxExpense")
    pretax_income  = av_series(inc_a, "incomeBeforeTax")

    # Balance sheet
    shares_diluted = av_series(bal_a, "commonStockSharesOutstanding")  # proxy if weighted average isn’t present
    equity         = av_series(bal_a, "totalShareholderEquity")
    debt_primary   = av_series_sum(bal_a, ["shortTermDebt", "longTermDebt"])
    debt_alt       = av_series_sum(bal_a, ["currentLongTermDebt", "longTermDebtNoncurrent"])
    total_debt     = debt_primary if (debt_primary is not None and not debt_primary.empty) else debt_alt
    if total_debt is None or total_debt.empty:
        total_debt = av_series(bal_a, "totalDebt")
    if total_debt is None or total_debt.empty:
        total_debt = av_series(bal_a, "shortLongTermDebtTotal")
    cash = av_series(bal_a, "cashAndCashEquivalentsAtCarryingValue")
    if cash.empty: cash = av_series(bal_a, "cashAndCashEquivalents")
    if cash.empty: cash = av_series(bal_a, "cashAndShortTermInvestments")

    # Cash flow
    cfo   = av_series(cfs_a, "operatingCashflow")
    capex = av_series(cfs_a, "capitalExpenditures")

    years = sorted(
        set(revenue.index) | set(net_income.index) | set(diluted_eps.index) |
        set(shares_diluted.index) | set(ebit.index) | set(tax_expense.index) |
        set(pretax_income.index) | set(equity.index) | set(total_debt.index) |
        set(cash.index) | set(cfo.index) | set(capex.index)
    )[-11:]

    if not years:
        raise RuntimeError("Alpha Vantage fundamentals contain no annual years. Try Yahoo/FMP or wait due to rate limits.")

    def A(s): return s.reindex(years).astype("float64")
    revenue, net_income, diluted_eps, shares_diluted, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex = [
        A(x) for x in [revenue, net_income, diluted_eps, shares_diluted, ebit, tax_expense, pretax_income, equity, total_debt, cash, cfo, capex]
    ]

    # EPS fallback if missing
    eps = diluted_eps.copy()
    if eps.isna().all() and not net_income.isna().all() and not shares_diluted.isna().all():
        with np.errstate(invalid="ignore", divide="ignore"):
            eps = net_income / shares_diluted.replace({0: np.nan})

    # FCF and ROIC
    fcf = (cfo - capex) if (not cfo.isna().all() and not capex.isna().all()) else pd.Series([np.nan]*len(years), index=years)
    with np.errstate(divide="ignore", invalid="ignore"):
        tax_rate = (tax_expense / pretax_income).clip(0, 1)
    nopat = (ebit * (1 - tax_rate.fillna(0.21))) if not ebit.isna().all() else net_income
    invested_capital = (total_debt.fillna(0) + equity.fillna(0) - cash.fillna(0)).replace({0: np.nan})
    with np.errstate(divide="ignore", invalid="ignore"):
        roic = (nopat / invested_capital).replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame({
        "Revenue": revenue, "EPS": eps, "Equity": equity, "FCF": fcf, "ROIC": roic, "SharesDiluted": shares_diluted
    }).sort_index().tail(11)

    # Diagnostics: how many annual rows AV gave per statement
    diag = {
        "annual_reports_counts": {
            "income": len(inc_a),
            "balance": len(bal_a),
            "cashflow": len(cfs_a)
        },
        "series_non_missing": {k:int(v) for k,v in df.notna().sum().to_dict().items()}
    }
    return df, years, "Alpha Vantage (patched)", diag
