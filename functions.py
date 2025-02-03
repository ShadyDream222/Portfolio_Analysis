#functions.py

import numpy as np
import pandas as pd 
import datetime as dt 
import pandas_datareader as web
import scipy.stats as stats

from datetime import datetime

import yfinance as yf 
import financedatabase as fd


import plotly.graph_objects as go
import plotly.express as px
import plotly as pt
import streamlit as st 
import plotly.io as pio

from jinja2 import Environment, FileSystemLoader
import base64

import logging 
#from scalene import scalene_profiler



#--------------------------------------------------------------------------------------------------------#
#IMPORT TICKER LIST
#--------------------------------------------------------------------------------------------------------#
@st.cache_data
def load_data()-> list :

    """
    Charge les données issus de financedatabase
    
    Parameters:
    -----------
    None
   
    Returns:
    --------
    list 
        Liste de l'ensemble des tickers de la database
    """
    
    try:
        ticker_list = pd.concat([fd.ETFs().select().reset_index()[['symbol','name']],
                                fd.Equities().select().reset_index()[['symbol','name']]])
        ticker_list = ticker_list[ticker_list['symbol'].notna()]
        ticker_list['symbol_name'] = ticker_list.symbol + '-' + ticker_list['name']
        logging.info(f'Liste de tous les tickers chargée avec succès')

    except (KeyError, AttributeError, ValueError) as e:
        logging.info(f'Erreur dans le chargement des listes de tickers: {e}')

    return ticker_list
#--------------------------------------------------------------------------------------------------------#





#--------------------------------------------------------------------------------------------------------#
#GET VALID TICKERS DATA
#--------------------------------------------------------------------------------------------------------#
@st.cache_data
def get_valid_tickers_data(selected_tickers_list , start_date , end_date) -> tuple:
    """
    Assure la disponibilité des tickers contenant des données viables
    
    Parameters:
    -----------
    selected_tickers_list : list
        liste des tickers choisis par l'utilisateur
    start_date : datetime.date
        Date de début de la collection des données
    end_date : datetime.date
        Date de fin 
        
    Returns:
    --------
    tuple (list, DataFrame)
       - Liste contenant les tickers valides parmis ceux choisis par l'utilisateur sur la période choisie
       - DataFrame contenant les données de ces tickers valides
    """
    valid_tickers = []
    all_data = []
    
    for ticker in selected_tickers_list:
        try:

            data = yf.download(ticker, 
                            start=start_date, 
                            end=end_date)['Close']
            
            
            #Check data not empty
            if not data.empty: 
                valid_tickers.append(ticker)
                data.name = ticker
                all_data.append(data)
                #tickers_data[ticker] = data  


            else:
                st.error(f"Aucune donnée trouvé pour {ticker} entre {start_date} et {end_date}. Non chargé")
                logging.warning(f"Aucune donnée trouvé pour {ticker} entre {start_date} et {end_date}. Non chargé")

        except (KeyError, ValueError, IndexError) as e:
            st.error(f"Erreur lors du téléchargement des données pour {ticker}")
            logging.exception(f"Erreur lors du téléchargement des données pour {ticker} : {e}")



        #df = pd.DataFrame(tickers_data) if valid_tickers else pd.DatFrame()
        df = pd.concat(all_data, axis = 1 ) if all_data else pd.DataFrame()

    
    return valid_tickers, df
#--------------------------------------------------------------------------------------------------------#





#--------------------------------------------------------------------------------------------------------#
#IMPORT LOGOS
#--------------------------------------------------------------------------------------------------------#
@st.cache_data
def fetch_logo(ticker:str) -> str:
    """
    Extrait les logos du ticker choisi du site  https://logo.clearbit.com/
    Parameters:
    -----------
   ticker : str
        Symbole du ticker
   
    Returns:
    --------
    str
        l'URL du logo 
    """
    try:
        url = yf.Ticker(ticker).info['website']
        if url:
            logo_url = 'https://logo.clearbit.com/' + url.replace('https://www.', '')
            logging.info("-"*20)
            logging.info(f"Logo chargé avec succès pour {ticker}.")
            return logo_url
        
        else :
            logging.warning(f"Aucune donnée trouvée pour {ticker}.")
            return 
        
    except (KeyError, AttributeError) as e:
        st.error(f"Erreur lors du chargement du logo pour {ticker}")
        logging.error(f"Erreur lors du chargement du logo pour {ticker} : {e}")

    return "https://via.placeholder.com/65" 
#--------------------------------------------------------------------------------------------------------#






#--------------------------------------------------------------------------------------------------------#
#IMPORT ECONOMIC DATA
#--------------------------------------------------------------------------------------------------------#
@st.cache_data
def load_economic_indicators(start_date:datetime.date, end_date:datetime.date) -> pd.DataFrame:

    """
    Charge les indicateurs économiques à partir de l'API, 
    les nettoyer et les transformer pour effectuer une analyse de corrélation
    
    Parameters:
    -----------
    start_date : datetime.date
        Date du début de la collection des données
    end_date : datetime.date
        Date de fin
        
    Returns:
    --------
    pd.DataFrame
        DataFrame contenant les données des indicateurs économiques sur la période chosie
    """

    try: 
        logging.info(f"Extraction des indicateurs économiques de {start_date} à {end_date}.")
        interest_rates = web.DataReader('IRLTLT01USM156N', #in %/year
                                        'fred', 
                                        start=start_date, 
                                        end=end_date
                                        )
        
        dbaa = web.DataReader('DBAA',  #in %/year 
                            'fred', 
                            start = start_date, 
                            end = end_date
                            )
        
        data_dollars = web.DataReader('DTWEXBGS', #no unit
                                    'fred', 
                                    start = start_date,
                                    end = end_date
                                    )
        
        gold = yf.download('GLD',   #dollars usd 
                        start=start_date, 
                        end=end_date)['Close'].reset_index().round(2)
        
        euro_stock = yf.download("^STOXX50E",   #points d'indice
                                start = start_date, 
                                end = end_date)['Close'].reset_index().round(3)
        
        spgsci = yf.download("^SPGSCI",   #point d'indice
                            start=start_date,
                            end = end_date)['Close'].reset_index().round(2)
        
        logging.info("Extraction des indicateurs économiques réussie")

    except (ValueError, KeyError, Exception) as e:
        st.error(f"Erreur dans l'extraction des données économiques")
        logging.error(f"Erreur dans l'extraction des données économiques : {e}")
    
    try : 
        logging.info("Fusion et transformation des indicateurs économiques")
        first_merge = pd.merge(dbaa, 
                            interest_rates, 
                            on = 'DATE', 
                            how = 'outer'
                            )


        second_merge = pd.merge(first_merge, 
                                data_dollars, 
                                on = 'DATE', 
                                how = "outer").reset_index()
        

        second_merge['IRLTLT01USM156N'] = second_merge['IRLTLT01USM156N'].ffill()


        #Si l'utilisateur prend une date qui n'est pas un 1er jour du mois, 
        #les premières valeurs sont manquantes jusqu'aux 1er jour du mois suivant car valeur seulement le premier jour du mois
        #Ici, calcule le premier jour du mois en question pour remplir les valeurs manquantes
        if start_date.day > 1:

            first_monthly_day = start_date.replace(day=1)

            try:
                # Extraction du taux au premier jour du mois
                value = web.DataReader('IRLTLT01USM156N', 'fred', start=first_monthly_day, end=first_monthly_day)['IRLTLT01USM156N'].iloc[0]
                second_merge['IRLTLT01USM156N'] = second_merge['IRLTLT01USM156N'].fillna(value) #Remplir avec la donnée du premier jour

                logging.info("Remplissage des valeurs manquantes du premier mois réussi")

            except (IndexError, KeyError) as e:
                st.error(f"Pas de données disponibles pour le {first_monthly_day}")
                logging.error(f"Pas de données disponibles pour le {first_monthly_day} : {e}")

        #Remplir NaN par moyenne sur une peridode de 30 jours
        second_merge['DBAA'] = second_merge['DBAA'].fillna(second_merge['DBAA'].rolling(window=30, min_periods=1).mean()).round(2).bfill() 
        second_merge["DTWEXBGS"] = second_merge["DTWEXBGS"].fillna(second_merge["DTWEXBGS"].rolling(window=30, min_periods=1).mean()).round(2).bfill()

        final_data = second_merge.copy()
        for df in [gold, euro_stock,spgsci]:

            final_data = pd.merge(final_data, 
                    df, 
                    left_on = 'DATE',
                    right_on = 'Date', 
                    how = 'inner' )

        final_data = final_data.rename(columns = {'DBAA': 'US rated BAA corporate bonds', 
                                                    'IRLTLT01USM156N':'US 10-year Treasury bonds',
                                                    'DTWEXBGS':'Dollar exchange rate index', 
                                                    #'Close_x' : 'Gold shares',
                                                    'GLD' : 'Gold shares',
                                                   # 'Close_y':'EuroStock index',
                                                    '^STOXX50E':'EuroStock index',
                                                   # 'Close':'SPGSCI',
                                                    '^SPGSCI':'SPGSCI'}
                                                    ).drop(columns=['DATE','Date_x','Date_y'])

        
        final_data = final_data[['Date'] + [col for col in final_data.columns if col != 'Date']]
        
        logging.info("Transformation des indicateurs économiques réussie")
        return final_data
        
    except (KeyError, ValueError, Exception) as e:
        st.error("Erreur dans la transformation des données économiques")
        logging.error(f"Erreur dans la transformation des données économiques : {e}")
#--------------------------------------------------------------------------------------------------------#





#--------------------------------------------------------------------------------------------------------#
#NORMALIZE DATA FOR CORRELATION 
#--------------------------------------------------------------------------------------------------------#
def normalize_data(data : pd.DataFrame) -> pd.DataFrame: 
    """
    Normalise les indicateurs économiques en les convertissant en variations en pourcentage
    Cette transformation permet une meilleure analyse de corrélation
    
    Parameters:
    -----------
    data : pd.DatFrame
        DataFrame contenant les indicateurs économiques nettoyés

    Returns:
    --------
    pd.DataFrame
        DataFrame contenant les indicateurs économiques normalisés
    """ 
    try:
        if data.empty:
            logging.warning("Le DataFrame Economic data est vide. Aucun traitement effectué")
            st.warning("Le DataFrame n'as pas pu être normalisé")
            return data 

        logging.info("Début de la normalisation des indicateurs économiques.")

        # Transformation en rendement pour les séries financières
        for col in ['Gold shares', 'EuroStock index', 'SPGSCI']:
            if col in data.columns:
                data[f'{col}_pct_daily'] = data[col].pct_change()

            else :
                st.warning(f"Colonne {col} non trouvée dans les données")
                logging.warning(f"Colonne {col} non trouvée dans les données")

        if 'Dollar exchange rate index' in data.columns:
            data['Dollar exchange rate index_pct_daily'] = data['Dollar exchange rate index'].pct_change()
        else:
            st.warning("Colonne 'Dollar exchange rate index' non trouvée dans les données")
            logging.warning("Colonne 'Dollar exchange rate index' non trouvée dans les données")

        data = data.dropna()
        data = data.reset_index(drop = True)

        logging.info("Normalisation des données économiques terminés avec succès")
        return data

    except (KeyError, ValueError, AttributeError) as e:
        st.error("Erreur dans la normalisation des données économiques")
        logging.error(f"Erreur dans la normalisation des données économiques : {e}")

        return pd.DataFrame()
#--------------------------------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------------------------------#
#CALCUL THE CORRELATION 
#--------------------------------------------------------------------------------------------------------#
def calculate_correlation(selected_asset : str, yfdata : pd.DataFrame,  normalized_economic_data : pd.DataFrame)  -> pd.DataFrame:
    """
    Calcule la corrélation entre l'actif choisi et les indicateurs économiques
    Calcule la p-value de Pearson (test statistique)
    
    Parameters:
    -----------

    selected asset : str
        le symbole de l'actif choisi pour le calcul des corrélations

    yfdata : pd.DatFrame
        DataFrame contenant les actifs du portefeuille

    normalized_economic_data : pd.DataFrame
        DataFrame contenant les données économiques normalisées 

    Returns:
    --------
    pd.DataFrame
        DataFrame contenant les résultats de corrélation 
        et des Pearson p_values avec un commentaire d'interpretation statistique
    """ 

    try:

        data_correlation_selected_asset = yfdata[yfdata['ticker'] == selected_asset]
        logging.info(f"Calcul de la corrélation pour: {selected_asset}")

        if data_correlation_selected_asset.empty:
            st.warning(f"Aucune donnée trouvé pour: {selected_asset}")
            logging.warning(f"Aucune donnée trouvé pour: {selected_asset}")
            return None

        data_correlation_selected_asset = data_correlation_selected_asset.dropna()

        data_correlation_selected_asset['Date'] = pd.to_datetime(data_correlation_selected_asset['Date']).dt.tz_localize(None)

        # Merge avec la data normalized
        merged_data = pd.merge(
            data_correlation_selected_asset[['Date', 'price_pct_daily']],
            normalized_economic_data,
            on='Date',
            how='inner'
        )

        # Merge de(s) actif(s) avec les indicateurs économiques
        merged_data_features = merged_data[['US rated BAA corporate bonds',
                                            'US 10-year Treasury bonds',
                                            'Dollar exchange rate index',
                                            'Gold shares_pct_daily', 
                                            'EuroStock index_pct_daily',
                                            'SPGSCI_pct_daily',
                                            'price_pct_daily']]
        

        # Matrice de corélation
        econ_corr = merged_data_features.corr()
        logging.info("Matrice de corrélation calculée avec succès")

        econ_corr_filtered = econ_corr[['price_pct_daily']].drop('price_pct_daily').rename(columns={'price_pct_daily': "Correlation"})

        # Calcul de la p-value de Pearson (test statistique)
        econ_corr_filtered["Pearson p-value"] = econ_corr_filtered.index.map(
            lambda col: stats.pearsonr(merged_data_features['price_pct_daily'], 
                                       merged_data_features[col])[1]
        )
        logging.info("Calcul du Pearson p-value avec succès")

        # Commentaire interpretatif  sur la p-value
        econ_corr_filtered["Comments"] = econ_corr_filtered["Pearson p-value"].apply(
            lambda p: "The correlation is statistically significant." if p < 0.05
            else "The correlation is not statistically significant."
        )
        
        st.table(econ_corr_filtered)

        return econ_corr_filtered
    
    except (Exception, KeyError) as e:
        st.error(f"le Dataset de {selected_asset} est vide pour faire une corrélation")
        logging.error(f"le Dataset de {selected_asset} est vide pour faire une corrélation : {e}")
        return None
#--------------------------------------------------------------------------------------------------------#

def calculate_all_correlations(financial_data : pd.DataFrame, normalized_economic_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule l'integralité des correlations des actifs avec les indicateurs économiques

    Parameters:
    -----------


    financial data : pd.DatFrame
        DataFrame contenants les données des actifs

    economic_data : pd.DataFrame
        DataFrame contenant les données des indicateurs économiques normalisées

        
    Returns:
    --------
    pd.DataFrame
        DataFrame contenant le tableau des corrélations
    """
    try:

        merged_df = pd.merge(financial_data[['Date', 'price_pct_daily','ticker']], normalized_economic_data, on='Date', how='inner')
        tickers = merged_df['ticker'].unique()
        indicators = [col for col in merged_df.columns if col not in ['Date', 
                                                                      'ticker', 
                                                                      'price_pct_daily',
                                                                      'Gold shares',
                                                                      'EuroStock Index',
                                                                      'SPGSCI']]

        correlation_matrix = pd.DataFrame(index=tickers, columns=indicators)

        for ticker in tickers:
            subset = merged_df[merged_df['ticker'] == ticker]
            for indicator in indicators:
                    correlation_matrix.loc[ticker, indicator] = subset['price_pct_daily'].corr(subset[indicator])

        return correlation_matrix

    except Exception as e:

        logging.error(f"Error in correlation calculation : {e}")
        st.error("Error in correlation calculation")
        return pd.DataFrame()



#--------------------------------------------------------------------------------------------------------#
#VISUALIZE ECONOMIC INDICATORS
#--------------------------------------------------------------------------------------------------------#
def visualize_economic_indicators(data : pd.DataFrame) -> None:
    """
    Visualise l'évolution de chaque indicateur économique sous forme de graphique

    Parameters:
    -----------
    data : pd.DatFrame
        DataFrame contenant les indicateurs économiques nettoyés
        
    Returns:
    --------
        None 
    """
    try:

        if data is None or data.empty:
            st.error("Aucune donnée pour la visualisation des données économiques")
            logging.error("Aucune donnée pour la visualisation des données économiques")
            return 
            
        
        st.subheader("Economic Indicators Trends")

        columns_to_plot = data.drop(columns = "Date", errors= "ignore").columns.tolist()

        if not columns_to_plot:
                st.error("Aucune colonne valide pour la visualisation des données économiques")
                logging.warning("Aucune colonne valide pour la visualisation des données économiques")

        rows = [columns_to_plot[i:i + 2] for i in range(0, len(columns_to_plot), 2)]
        
        for row in rows:
            cols = st.columns(2)
            for i, column in enumerate(row):
                with cols[i]: 
                    fig = px.line(data, 
                                x = 'Date', 
                                y = column, 
                                title = column
                                )
                    
                    fig.update_layout(xaxis_title = "Date", 
                                    yaxis_title = column, 
                                    template = "plotly_dark"
                                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

    except (KeyError, ValueError, AttributeError) as e:
        st.error("Erreur dans la génération des graphiques des indicateurs économiques")
        logging.exception(f"Erreur lors de la visualisation des indicateurs économiques : {e}")
#--------------------------------------------------------------------------------------------------------#




#--------------------------------------------------------------------------------------------------------#
#ESG VERIFICATION
#--------------------------------------------------------------------------------------------------------#
@st.cache_data
def fetch_esg_data(ticker:str) -> pd.DataFrame:

    """
    Extrait les données ESG complètes du ticker choisi depuis yfinance
    
    Parameters:
    -----------
    ticker : str
        Le symbole du ticker

    Returns:
    --------
    pd.DataFrame
        DataFrame contenant les données ESG pour le ticker sélectionné
    """
    
    try:
        logging.info(f"Extraction des données ESG pour {ticker}")
        info = yf.Ticker(ticker).sustainability
        
        if info is not None and not info.empty:
            df = info.T
            df['Ticker'] = str(ticker)
            logging.info(f"Données ESG récupérées avec succès pour {ticker}")
            return df 

        else:
            st.warning(f"Aucune donnée ESG disponible pour {ticker}")
            logging.warning(f"Aucune donnée ESG disponible pour {ticker}")
        
    except (KeyError, AttributeError) as e:
        logging.exception(f"Erreur lors de la récupération des données ESG pour {ticker} : {e}")
        pass

    return pd.DataFrame()  
#--------------------------------------------------------------------------------------------------------#




#--------------------------------------------------------------------------------------------------------#
# CLEAN ESG DATA 
#--------------------------------------------------------------------------------------------------------#
def clean_esg_data(data : pd.DataFrame) -> pd.DataFrame:
            
    """
    Nettoie et transforme les données ESG en extrayant 
    les valeurs des colonnes contenant des dictionnaires
    
    Parameters:
    -----------
    ticker : pd.DataFrame
        DataFrame contenant les données ESG brutes

    Returns:
    --------
    pd.DataFrame
        DataFrame contenant les données ESG nettoyées
    """
    try :
        if data.empty:
            st.warning("Le DataFrame ESG est vide. Aucun traitement effectué")
            logging.warning("Le DataFrame ESG fourni est vide. Aucun traitement effectué")
            return data

        colonnes = [
                        "peerEsgScorePerformance",
                        "peerGovernancePerformance",
                        "peerSocialPerformance",
                        "peerEnvironmentPerformance",
                        "peerHighestControversyPerformance"
                        ]
        
        esg_new_data = pd.DataFrame()

        for col in colonnes:
            if col in data.columns:
                expanded_cols = pd.DataFrame(data[col].tolist(), 
                                            index=data.index)
                
                expanded_cols = expanded_cols.rename(columns={
                    "min": f"{col} - Minimum Score",
                    "avg": f"{col} - Average Score",
                    "max": f"{col} - Maximum Score"
                }).round(2)
                data = pd.concat([data, expanded_cols], 
                                axis=1)
            else :
                st.warning(f"Colonne {col} non trouvée dans les données ESG")
                logging.warning(f"Colonne {col} non trouvée dans les données ESG")
            
                
        esg_new_data = data.drop(columns=[col for col in colonnes if col in data.columns])

        if 'ratingYear' in esg_new_data.columns:
            esg_new_data['ratingYear'] = esg_new_data['ratingYear'].astype(str)

        if 'esgPerformance' in esg_new_data.columns:
            esg_new_data['ESG Status'] = esg_new_data['esgPerformance'].map({
                'LAG_PERF': "Deceiving Performance",
                'AVG_PERF': "Deceiving Performance"
            }).fillna("Satisfying Performance")
        else:
            logging.warning("Colonne 'esgPerformance' non trouvée dans les données ESG.")

        logging.info("Nettoyage et transformation des données ESG terminés avec succès")
        return esg_new_data

    except (KeyError, ValueError, AttributeError, TypeError) as e:
        st.exception(f"Erreur lors du nettoyage et de la transformation des données ESG : {e}")
        logging.exception("Erreur lors du nettoyage et de la transformation des données ESG")
        return pd.DataFrame()
#--------------------------------------------------------------------------------------------------------#





#--------------------------------------------------------------------------------------------------------#
#CALCUL OF SHARPE RATIO
#--------------------------------------------------------------------------------------------------------#
def calculate_sharpe_ratio(data: pd.DataFrame, risk_free_rate: float = .01) -> float:  #Annualized
    """
    Calcule le Sharpe Ratio annualisé du portefeuille
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame contenant les prix des actifs
    risk_free_rate : float, optionnel
        Taux sans risque (par défaut 0.01)
    
    Returns:
    --------
    float
        Sharpe Ratio
    """
    try :
        if data.empty:
            st.warning("Le DataFrame est vide. Impossible de calculer le Sharpe Ratio")
            logging.warning("Le DataFrame est vide. Impossible de calculer le ratio de Sharpe")
            return -1

        data = data.pivot(index="Date", 
                        columns='ticker', 
                        values='price'
                        ).dropna()

        log_returns = np.log(1 + data.pct_change())

        random_weights = np.random.random(len(data.columns))
        normalized_weights = random_weights / np.sum(random_weights)

        expected_return = np.sum(log_returns.mean() * normalized_weights) * 252
        expected_vol = np.sqrt(
            np.dot(
                normalized_weights.T,
                np.dot(
                    log_returns.cov() * 252,
                    normalized_weights
                )
            )
        )
    
        sharpe_ratio = round((expected_return - risk_free_rate) / expected_vol,2)
        logging.info(f"Sharpe Ratio calculé avec succès")
        return sharpe_ratio

    except (KeyError, ValueError, ZeroDivisionError, AttributeError) as e:
        st.exception("Erreur lors du calcul du Sharpe Ratio")
        logging.exception(f"Erreur lors du calcul du Sharpe Ratio : {e}")
        return -1
#--------------------------------------------------------------------------------------------------------#






#--------------------------------------------------------------------------------------------------------#
#CALCUL OF HISTORICAL VAR
#--------------------------------------------------------------------------------------------------------#
def calcul_historic_VaR(data : pd.DataFrame, level_confidence : float) -> float:
    """
    Calcule la VaR historique d'un portefeuille
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame contenant les prix des actifs
    level_confidence : float
        Niveau de confiance pour le calcul de la VaR (Généralement 0.95 ou 0.99)
    
    Returns:
    --------
    float
        VaR Historique 
    """
    try : 
        data = data.pivot(index="Date", 
                        columns='ticker', 
                        values='price'
                        ).dropna()


        returns = data.pct_change()
        returns = returns.dropna()

        return np.percentile(returns, (1 - level_confidence) * 100).round(2)

    except (KeyError, ValueError, IndexError, AttributeError) as e:
        st.exception("Erreur lors du calcul de la VaR historique")
        logging.exception(f"Erreur lors du calcul de la VaR historique : {e}")

        return -1
#--------------------------------------------------------------------------------------------------------#






#--------------------------------------------------------------------------------------------------------#
#CALCUL OF MAX DRAWDOWN
#--------------------------------------------------------------------------------------------------------#
def calculate_max_drawdown(data : pd.DataFrame) -> tuple:
    """
    Calcule le drawdown maximal d'un portefeuille
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame contenant les rendements
    
    Returns:
    --------
    tuple (pd.DataFrame, float)
        - DataFrame mis à jour avec le drawdown
        - Valeur du drawdown maximal
    """
    try:
        if data.empty:
            logging.warning("Le DataFrame fourni est vide. Impossible de calculer le drawdown.")
            return data, -1

        data = data.dropna()

        data = data.copy()
        data.loc[:, 'Cumulative Returns'] = (1 + data['price_pct_daily']).cumprod()
        data.loc[:, 'Cumulative_max'] = data['Cumulative Returns'].cummax()
        data.loc[:, 'Drawdown'] = (data['Cumulative Returns'] - data['Cumulative_max']) / data['Cumulative_max']

        mdd = data['Drawdown'].min()

        logging.info(f"Drawdown maximal calculé avec succès")

        return data, mdd

    except (KeyError, ValueError, IndexError, AttributeError) as e:
        logging.exception(f"Erreur lors du calcul du drawdown maximal : {e}")
        return data, -1
#--------------------------------------------------------------------------------------------------------#






#--------------------------------------------------------------------------------------------------------#
#HTML REPORT GENERATOR
#--------------------------------------------------------------------------------------------------------#
def generate_html_report(yfdata : pd.DataFrame, start_date : dt.datetime , end_date : dt.datetime, esg_data_cleaned : pd.DataFrame, all_correlations_table : pd.DataFrame ) -> str:
    """
    Génère un rapport HTML contenant les performances du portefeuille, les scores ESG 
    et autres informations jugées essentielles
    
    Parameters:
    -----------
    yfdata : pd.DataFrame
        Données des actifs du portefeuille
    start_date : datetime
        Date de début d'analyse
    end_date : datetime
        Date de fin d'analyse
    esg_data_cleaned : pd.DataFrame
        Données ESG nettoyées pour les actifs du portefeuille
    
    Returns:
    --------
    str
        Rapport HTML formaté 
    """
    try:

        report_date = datetime.now().strftime("%Y-%m-%d")
        author_name = "GUIDDIR Lucas"
        tickers = yfdata['ticker'].unique().tolist()


        #Graphique portfolio performance 
        try:
                yfdata_performance = yfdata.pivot(index="Date", 
                        columns='ticker', 
                        values='price_pct_daily'
                        ).dropna()
                
                yfdata_performance['Means returns'] = yfdata_performance[tickers].mean(axis = 1)
                yfdata_performance = yfdata_performance.reset_index()

                portfolio_performance_fig = px.line(yfdata_performance, 
                                                x = 'Date', 
                                                y = 'Means returns', 
                                                #color = 'ticker', 
                                                title = 'Portfolio Performance',
                                                markers= True,
                                                color_discrete_sequence=px.colors.qualitative.Plotly
                                                )
                
                portfolio_performance_chart = pio.to_image(portfolio_performance_fig, 
                                                        format = 'png')
    
                portfolio_performance_chart_base64 = base64.b64encode(portfolio_performance_chart).decode('utf-8')
                

        except  Exception as e:
            logging.exception(f"Erreur lors de la création du graphique de performance de portefeuille : {e}")
            portfolio_performance_chart_base64 = ""



        #Graphique de performance individuelle
        try:
            with open('images/individual_performance_chart_base64.png', 'rb') as img_file:
                individual_performance_chart_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        except:
            st.warning("L'image 'individual_performance.png' est introuvable")
            logging.warning("Image 'indivudual_performance.png' introuvable")
            individual_performance_chart_base64 = ""


        #Graph Matrice de corrélation
        try :
            with open('images/correlation_matrix.png', 'rb') as img_file:
                correlation_matrix_chart_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        except:
            st.warning("L'image 'correlation_matrix.png' est introuvable")
            logging.warning(f"L'image 'correlation_matrix.png' est introuvable :{e}")
            correlation_matrix_chart_base64 = ""


        #Tableau de corrélation 
        try :
            all_correlations_table = all_correlations_table.reset_index().rename(columns = {'index': 'ticker'})
            corr_data = {}
            for ticker in tickers:
                ind_row = all_correlations_table[all_correlations_table['ticker'] == ticker]
                if not ind_row.empty:
                    corr_data[ticker] = {
                'CB' : round(ind_row['US rated BAA corporate bonds'].values[0],4), #.values[0],
                'TB' :  round(ind_row['US 10-year Treasury bonds'].values[0],4), #.values[0],
                'ER' : round(ind_row['Dollar exchange rate index'].values[0],4), #.values[0],
                'GS': round(ind_row['Gold shares_pct_daily'].values[0],4), #.values[0],
                'ES'  : round(ind_row['EuroStock index_pct_daily'].values[0],4), #.values[0],
                'SPGSCI' : round(ind_row['SPGSCI_pct_daily'].values[0],4) #.values[0] 
                }
        except Exception as e:
            logging.error(f"Erreur dans la génération du tableau de corrélation : {e} ")
            st.error("Erreur dans la génération du tableau de corrélation" )


        # Données ESG
        try : 
            esg_data = {}
            for ticker in tickers:
                esg_row = esg_data_cleaned[esg_data_cleaned['Ticker'] == ticker]
                if not esg_row.empty:
                    esg_data[ticker] = {
                        'Environment': esg_row['Environment Score'].values[0],
                        'Social': esg_row['Social Score'].values[0],
                        'Governance': esg_row['Governance Score'].values[0],
                        'ESG' : esg_row['ESG Score'].values[0],
                        'Status' : esg_row['ESG Status'].values[0]
                    }


            if not esg_data_cleaned.empty:
                esg_data_short = esg_data_cleaned[['Ticker', 'ESG Score']].reset_index()
                dunce_ticker = esg_data_short.loc[esg_data_short['ESG Score'].idxmin(), 'Ticker']
            else:
                dunce_ticker = "N/A"

        except Exception as e:
            logging.error(f"Erreur dans la génération du tableau ESG data {e}" )
            st.error("Erreur dans la génération du tableau ESG data")

        # Tentative de conclusion personnalisé 
        conclusion = f"The portfolio shows a strong performance with some areas for improvement in ESG scores. We need to stay focus on {dunce_ticker} which has the worst ESG Score"

        # Remplissage du modèle
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template('report_template.html')
        html_output = template.render(
            author_name = author_name,
            report_date = report_date,
            start_date = start_date,
            end_date = end_date,
            tickers = tickers,
            portfolio_performance_chart = f"data:image/png;base64,{portfolio_performance_chart_base64}",
            individual_performance_chart = f"data:image/png;base64,{individual_performance_chart_base64}",
            correlation_matrix_chart = f"data:image/png;base64,{correlation_matrix_chart_base64}",
            all_correlations_table = corr_data,
            esg_data = esg_data,
            conclusion = conclusion
        )
        logging.info("Rapport HTML généré avec succès")
        return html_output

    except Exception as e:
        st.exception("Erreur lors de la génération du rapport HTML")
        logging.exception(f"Erreur lors de la génération du rapport HTML : {e}")
        return ""

#--------------------------------------------------------------------------------------------------------#