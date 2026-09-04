# NIFTY Institutional Market Dashboard

This combines the uploaded notebooks into one Streamlit dashboard:

- IV Rank / IV Percentile
- Daily Expected Move
- Parkinson vs Close-to-Close Volatility
- Volatility Cone
- 60-day Hurst Regime
- Volatility Risk Premium (VRP)
- NIFTY vs BANK NIFTY 20-day Correlation
- 15-minute Liquidity Sweep Detector
- Optional live NIFTY Options OI Profile using Zerodha Kite

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The OI tab requires your own Zerodha Kite API key and access token. No credentials are stored in this project.

## Important
The dashboard is a statistical/educational decision-support tool, not investment advice.
