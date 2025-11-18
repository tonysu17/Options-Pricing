# utils/data_handler.py
import yfinance as yf
import pandas as pd
import datetime as dt

def fetch_option_chain(ticker_str):
    """ Fetches the entire option chain for a given ticker. """
    ticker = yf.Ticker(ticker_str)
    expirations = ticker.options
    
    chain = pd.DataFrame()
    for expiry in expirations:
        opt = ticker.option_chain(expiry)
        calls = opt.calls
        calls = 'call'
        puts = opt.puts
        puts = 'put'
        
        df = pd.concat([calls, puts])
        df = pd.to_datetime(expiry)
        chain = pd.concat([chain, df])
        
    return chain

def clean_option_data(chain, S0):
    """ Cleans and filters the raw option chain data. """
    # Calculate days to expiration
    chain = (chain - dt.datetime.now()).dt.days
    chain = chain / 365.0

    # Calculate mid price
    chain['mid_price'] = (chain['bid'] + chain['ask']) / 2.0
    
    # Filter out contracts with low liquidity or wide spreads
    chain = chain[chain['openInterest'] > 10]
    chain = chain[chain['volume'] > 5]
    chain = chain[chain['ask'] > 0]
    chain = chain[chain['bid'] > 0]
    
    # Filter out deep ITM/OTM options
    chain = chain[(chain['strike'] > 0.7 * S0) & (chain['strike'] < 1.3 * S0)]
    
    return chain