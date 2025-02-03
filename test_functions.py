#test_functions.py

import numpy as np
import pandas as pd
import yfinance as yf
import time
from functions import *


START_DATE =  pd.to_datetime("2024-01-01", format='%Y-%m-%d')
END_DATE = pd.to_datetime("2024-12-31", format='%Y-%m-%d')

YFDATA = yf.download("AAPL", START_DATE, END_DATE)
YFDATA = YFDATA['Close'].reset_index().melt(
                                    id_vars = ['Date'],    
                                    var_name = 'ticker', 
                                    value_name = 'price'
                                    )

YFDATA['price_start'] = YFDATA.groupby('ticker')['price'].transform('first')
YFDATA['price_pct_daily'] = YFDATA.groupby('ticker')['price'].pct_change().dropna()
YFDATA['price_pct'] = (YFDATA['price'] - YFDATA['price_start']) / YFDATA['price_start']


#--------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- TEST LOAD DATA  -------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------#

def test_load_data():
    """Check if load data returns viable tickers"""
    data = load_data()
    assert not data.empty, "La liste des tickers ne doit pas être vide"
    assert "symbol" in data.columns and "name" in data.columns, "Les colonnes symbol et name doivent exister"


def test_load_data_performance():
    """Test speed of load_data() """
    start_time = time.time()
    load_data()
    assert time.time() - start_time < 3, "Temps de chargement supérieur à 3s"
#--------------------------------------------------------------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- TEST VALID TICKERS ----------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------#

def test_get_valid_tickers_data():
    """Test la récupération des tickers valides avec des données simulées."""
    
    valid_tickers, data = get_valid_tickers_data(["AAPL", "INVALID"], START_DATE, END_DATE)
    
    assert valid_tickers == ["AAPL"], "Seul AAPL devrait être valide"
    assert isinstance(data, pd.DataFrame), "Le résultat doit être un DataFrame"
    assert not data.empty, "Le DataFrame ne doit pas être vide"
#--------------------------------------------------------------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- TEST LOGOS ------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------#

def test_fetch_logo():
    """Check if fetch logo returns a good logo"""
    logo_url = fetch_logo("AAPL")
    assert isinstance(logo_url, str), "L'URL doit être une chaîne"
    assert logo_url.startswith("http"), "L'URL doit commencer par http"


def test_fetch_logo_success():
    """Check if it returns the good website"""
    logo_url = fetch_logo("AAPL")
    assert logo_url.startswith("https://logo.clearbit.com/")
    assert "apple.com" in logo_url 

def test_fetch_logo_failure():
    """Check if it returns the default URL"""
    logo_url = fetch_logo(None)
    assert logo_url == "https://via.placeholder.com/65" 
#--------------------------------------------------------------------------------------------------------------------------------------#




#--------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------- TEST ESG DATA ---------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------#

def test_fetch_esg_data_success():
    """Check if relevant ESG columns exist """
    esg_data = fetch_esg_data("AAPL")
    expected_columns = {"totalEsg", "environmentScore", "socialScore", "governanceScore"}
    assert isinstance(esg_data, pd.DataFrame)  
    assert not esg_data.empty  
    assert expected_columns.issubset(esg_data.columns) # Vérifie que les colonnes existent

def test_fetch_esg_data_failure():
    """Check if invalid input returns an empty data"""
    esg_data = fetch_esg_data("INVALIDTICKER")
    assert esg_data.empty  
    esg_data = fetch_esg_data(None)
    assert esg_data.empty  
#--------------------------------------------------------------------------------------------------------------------------------------#





#--------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- TEST ECONOMIC INDICATORS ----------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------#

def load_economic_indicators_success():
    """Check if all data exists and not empty"""
    data = load_economic_indicators(START_DATE, END_DATE)
    assert not all(data).empty
    assert not data['US rated BAA corporate bonds'].any().isna()
    assert 'US rated BAA corporate bonds' in data.columns
    assert 'US 10-year Treasury bonds' in data.columns
    assert 'Dollar exchange rate index' in data.columns
    assert 'Gold shares' in data.columns
    assert 'EuroStock index' in data.columns
    assert 'SPGSCI' in data.columns


def test_load_economic_indicators_short_period():
    """Check if I have data for only 3 days"""
    start_date = pd.to_datetime("2024-01-01")
    end_date = pd.to_datetime("2024-01-04")  # 3 days
    data = load_economic_indicators(start_date, end_date)
    assert not data.empty
#--------------------------------------------------------------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- TEST NORMALIZE DATA ---------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------#

def test_normalize_data():
    """Check if all data has been normalized and not empty"""
    data = load_economic_indicators(START_DATE, END_DATE)
    normalized_data = normalize_data(data)
    assert not normalized_data.empty
    assert 'Gold shares_pct_daily' in normalized_data.columns  
    assert 'EuroStock index_pct_daily' in normalized_data.columns
    assert 'SPGSCI_pct_daily' in normalized_data.columns
    assert 'Dollar exchange rate index_pct_daily' in normalized_data.columns
#--------------------------------------------------------------------------------------------------------------------------------------#




#--------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------- TEST SHARPE RATIO ------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------#

def test_calculate_sharpe_ratio():
    """Test Sharpe ratio calculation"""
    sharpe = calculate_sharpe_ratio(YFDATA)
    assert isinstance(sharpe, float)

    data = pd.DataFrame()
    sharpe = calculate_sharpe_ratio(data)
    assert sharpe == -1
#--------------------------------------------------------------------------------------------------------------------------------------#




#--------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------ TEST HISTORIC VAR -------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------#

def test_calcul_historique_VaR():
    """Test VaR calculation"""
    var = calcul_historic_VaR(YFDATA, 0.95)
    assert isinstance(var, float)
    assert var < 0  

    data = pd.DataFrame()
    var = calcul_historic_VaR(data, 0.95)
    #assert var == float('nan')
    assert var == -1

    var = calcul_historic_VaR(YFDATA, 5)  #> 1
    assert var == -1
#--------------------------------------------------------------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------- TEST MAX DRAWDOWN  -------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------------#

def test_calculate_max_drawdown():
    """Test max drawdown calculation"""
    _, mdd = calculate_max_drawdown(YFDATA)
    assert isinstance(mdd, float)

    data = pd.DataFrame()
    _,mdd = calculate_max_drawdown(data)
    assert mdd == -1
#--------------------------------------------------------------------------------------------------------------------------------------#






