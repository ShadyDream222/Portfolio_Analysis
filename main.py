#main.py

import numpy as np
import pandas as pd 
import datetime as dt 
import pandas_datareader as web

from datetime import datetime
import scipy.stats as stats

import yfinance as yf 
import financedatabase as fd

import plotly.graph_objects as go
import plotly.express as px
import plotly as pt
import streamlit as st 
import plotly.io as pio

import logging 
from importlib import reload 
#from scalene import scalene_profiler

import base64
import os


if not os.path.exists('images'):
    os.makedirs('images')

st.set_page_config(
    layout = "wide",
    initial_sidebar_state="expanded")

import functions as fn 

reload(logging)

logging.basicConfig(
    level = logging.INFO, 
    format = '%(asctime)s | %(levelname)s : %(message)s',  
    filename = 'portfolio_analysis.log',
    filemode ='a')

st.title("Portfolio analysis")

with st.sidebar:
    ticker_list = fn.load_data()
    sel_tickers = st.multiselect('Portfolio Builder', 
                                 placeholder = 'Search tickers', 
                                 options = ticker_list['symbol_name'])
    
    sel_tickers_list = ticker_list[ticker_list['symbol_name'].isin(sel_tickers)]['symbol']

    cols = st.columns(4)

    for i, ticker in enumerate(sel_tickers_list): 
        try:
          cols[ i % 4 ].image(fn.fetch_logo(ticker), width = 65)

        except:
            cols [i % 4].subheader(ticker)
 

    #Date selector : 
    cols = st.columns(2)
    sel_dt1 = cols[0].date_input('Start Date', 
                                 value = dt.datetime(2024,1,1), 
                                 format = 'YYYY-MM-DD')
    
    sel_dt2 = cols[1].date_input('End Date', 
                                 format = 'YYYY-MM-DD')

    
    generate_btn = st.sidebar.button("Generate HTML Report")

    if sel_dt2 < sel_dt1:
        st.error("La End date est plus grande que la Start Date, choissisez une période convenable")
        logging.warning("Erreur dans le choix des dates")

    elif sel_dt2 == sel_dt1:
        st.error("Impossible de prendre 1 jour, prenez une période plus large")
        logging.warning("Erreur dans le choix des dates")

    else :
        pass


    if len(sel_tickers) != 0:
        
        try :

            valid_sel_tickers, yfdata = fn.get_valid_tickers_data(sel_tickers_list, 
                                                    start_date = sel_dt1,
                                                    end_date = sel_dt2)
            

            yfdata = yfdata.reset_index().melt(
                                                id_vars = ['Date'],    
                                                var_name = 'ticker', 
                                                value_name = 'price'
                                                )

            yfdata['price_start'] = yfdata.groupby('ticker')['price'].transform('first')
            yfdata['price_pct_daily'] = yfdata.groupby('ticker')['price'].pct_change().dropna()
            yfdata['price_pct'] = (yfdata['price'] - yfdata['price_start']) / yfdata['price_start']

        except Exception as e : 

            st.error("Erreur lors de la récupération des données des actifs")
            logging.error(f"Erreur d'extraction des données API: {e}")




#Tabs 
#-----------------------------------------------------------------------
tab1,tab2,tab3,tab4,tab5 = st.tabs(['Portfolio','Statistics', 'Key Metrics', 'ESG Data','Invest without Carbon'])

if len(sel_tickers) == 0:
    st.info('Select tickers to view plots')
else :
    st.empty()

#----------------------------------------- TAB 1 : PORTFOLIO OVERVIEW  -------------------------------------------------------------------#

    with tab1:
        st.subheader("Portfolio Overview")

        try: 

            tab1_cols = st.columns(3)
            fig = px.line(
            yfdata, 
            x = 'Date', 
            y ='price_pct', 
            color = 'ticker',
            markers = True,
            title = 'All Stocks',
            color_discrete_sequence=px.colors.qualitative.Plotly
            )

            fig.add_hline(y=0,line_dash = "dash", line_color = "white") 

            fig.update_layout(
            xaxis_title = None, 
            yaxis_title = "Performance (in %)",
            yaxis_tickformat = ',.0%',
            legend_title = "Tickers",
            )

            st.plotly_chart(fig, use_container_width=True)
            individual_performance_chart = pio.write_image(fig, 'images/individual_performance_chart_base64.png')#,scale=1, width=1200, height=800)
            #individual_performance_chart_base64 = base64.b64encode(individual_performance_chart).decode('utf-8')

        except Exception as e:
              st.error("Erreur de visualisation pour le portfolio performance")
              logging.info(f"Erreur de visualisation pour le portfolio performance : {e}")

        try: 

            portfolio_sharpe_ratio = fn.calculate_sharpe_ratio(yfdata)
            portfolio_VaR = fn.calcul_historic_VaR(yfdata,0.95) #à 95%
        
        except:
            st.error("Problème dans le calcul du Sharpe Ratio ou de la VaR")
            logging.warning(f"Problème dans le calcul du Sharpe Ratio de la VaR")

        tab1_cols[0].metric(label = 'Sharpe Ratio', 
                                         value = portfolio_sharpe_ratio)
                
        tab1_cols[1].metric(label = 'VaR',
                                         value = portfolio_VaR )

        #Individual stockplot 
        st.subheader("Individual stock")
        cols = st.columns(3)
        for i, ticker in enumerate(valid_sel_tickers): 

            try:

                cols[ i % 3 ].image(fn.fetch_logo(ticker), width = 65)

            except:
                cols [i % 3].subheader(ticker)


            #Stock Metrics
            cols2 = cols[i % 3].columns(3)

            cols2[0].metric(label = '50 days average', 
                            value = round(yfdata[yfdata['ticker'] == ticker ]['price'].tail(50).mean(),1))
            cols2[1].metric(label = '1 Year Low', 
                            value = round(yfdata[yfdata['ticker'] == ticker ]['price'].tail(365).min(),1))
            cols2[2].metric(label = '1 Year High', 
                            value = round(yfdata[yfdata['ticker'] == ticker ]['price'].tail(365).max(),1))
        
            #Stock plot
            fig = px.line(
            yfdata[yfdata["ticker"] == ticker], 
            x = 'Date', 
            y = 'price', 
            #y = 'price_pct',
            markers = True )

            fig.update_layout(xaxis_title= None, yaxis_title =None)
            cols[i % 3].plotly_chart(fig,use_container_width=True,)

#----------------------------------------------------- TAB 2 : STATISTICS  ---------------------------------------------------#
    with tab2:

        #Le choix des indicateurs économiques a notamment été choisi en s'inspirant de cet article scientifique :
        # Cross-asset relations, correlations and economic implications, David G. McMillan
        # https://www.sciencedirect.com/science/article/abs/pii/S1044028318302084

        st.subheader("Correlation between stocks")
        if sel_tickers :

            returns = yfdata[['Date','ticker','price_pct']].pivot(index='Date', columns='ticker', values='price_pct').dropna()
            corr = returns.corr()
            fig_corr = px.imshow(corr, 
                                text_auto = True, 
                                color_continuous_scale = 'magma', 
                                title = 'Correlation Heatmap'
                                )
            
            st.plotly_chart(fig_corr, use_container_width=True)
            pio.write_image(fig_corr, 'images/correlation_matrix.png')

        else :
            st.info("Select at least one ticker to view the correlation matrix.")


        economic_data = fn.load_economic_indicators(start_date = sel_dt1,
                                                    end_date = sel_dt2)
        

        st.subheader("Trends and Analysis")

        st.write("Economic Data", economic_data)

        fn.visualize_economic_indicators(economic_data)


        st.subheader("Correlation with Economic Indicators")

        normalized_economic_data = fn.normalize_data(economic_data)
        normalized_economic_data = normalized_economic_data.dropna()

        all_correlations_table = fn.calculate_all_correlations(yfdata, normalized_economic_data)
        #st.write("All correlation table", all_correlations_table)


        selected_asset = st.selectbox("Select an Asset", 
                                    options = valid_sel_tickers, 
                                    help = "Choose an asset to analyze its correlation with economic indicators.")
        
        if selected_asset:

            try : 
                
                econ_corr_filtered = fn.calculate_correlation(selected_asset, 
                                                              yfdata, 
                                                              normalized_economic_data)
        

                fig_econ_corr = px.bar(
                    econ_corr_filtered,
                    x = econ_corr_filtered[['Correlation']].index,
                    y = 'Correlation',
                    color = "Correlation",
                    color_continuous_scale = "magma",
                    title = f"Correlation of {selected_asset} with Economic Indicators"
                )
                st.plotly_chart(fig_econ_corr, use_container_width=True)

                    

            except Exception as e : 
                logging.error(f"Une erreur est survenue dans le calcul des corrélations {e}")

        else:
            st.info("Select an asset to view its correlation with economic indicators.")


#---------------------------------------- TAB 3 : KEY METRICS OVERVIEW -----------------------------------------------------#


    with tab3:
        st.subheader("Key Metrics Overview")

        selected_asset = st.selectbox("Select an Asset", 
                                    options=valid_sel_tickers, 
                                    help="Choose an asset to see performance overview")
        if selected_asset:
            try : 

                st.image(fn.fetch_logo(selected_asset), width=65)

                tab5_cols = st.columns(3)

                metrics_data = yfdata[yfdata['ticker'] == selected_asset]
                metrics_data = metrics_data.dropna()

                try : 

                    sharpe_ratio = fn.calculate_sharpe_ratio(metrics_data)

                    VaR = fn.calcul_historic_VaR(metrics_data,
                                              level_confidence = 0.95)
                    
                    metrics_data, mdd = fn.calculate_max_drawdown(metrics_data)
                
                    
                except Exception as e:
                    st.error(f"Error dans le calcul du Sharpe Ratio, de la VaR ou du MDD : {e}")


                tab5_cols[0].metric(label = 'Sharpe Ratio', 
                                         value = sharpe_ratio)
                
                tab5_cols[1].metric(label = 'VaR',
                                         value = VaR )
                
                tab5_cols[2].metric(label = 'MDD',
                                         value =f"{round(mdd * 100, 2)}%")
                

                if VaR > 0:
                    st.error("Attention : VaR positive, Possible Erreur")
                

                metrics_data['SMA 30 days'] = metrics_data['price'].rolling(window=30).mean()  #Short term trends : 30 days
                metrics_data['SMA 90 days'] = metrics_data['price'].rolling(window=90).mean()  #long term trends : 90 days
                
                
                st.write("Metrics Data", metrics_data)

                st.subheader("Metric Analysis")

                if not metrics_data.empty:

                    fig_ma = px.line(
                    metrics_data, 
                    x='Date', 
                    y=['price', 'SMA 30 days', 'SMA 90 days'], 
                    title=f"Moving Average for {selected_asset}",
                labels={"value": "Price", "variable": "Type"}
                )

                    fig_ma.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Price",
                        legend_title="Metrics",
                        template="plotly_white")
                    
                    st.plotly_chart(fig_ma, use_container_width=True)


                    metrics_data["Volatility"] = metrics_data["price_pct"].rolling(window = 30).std() * np.sqrt(252).round(2)

                    fig_vol = px.line(
                    metrics_data, 
                    x='Date', 
                    y='Volatility', 
                    title=f"Volatility over a 30 days period for {selected_asset}", 
                labels={"value": "Volatility", "variable": "Type"},
                markers= True
                )
                    
                    fig_vol.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Volatiliy",
                        legend_title="Metrics",
                        template="plotly_white"
                        )
                    
                    st.plotly_chart(fig_vol, use_container_width=True)


                    fig_dd = px.line(metrics_data, 
                                     x='Date', 
                                     y='Drawdown', 
                                     title=f'Drawdown for {selected_asset}',
                                     markers= True
                                     )

                    fig_dd.update_layout(xaxis_title='Date',
                                      yaxis_tickformat='.2%',
                                      yaxis_title='Drawdown (%)',
                                      legend_title="Metrics",
                                      )
                    
                    fig_dd.add_hline(y=0,line_dash = "dash", line_color = "white") 
                    
                    st.plotly_chart(fig_dd, use_container_width=True)


                else :
                    st.info("Pas assez de données pour plot les moving arverage")

            

            except Exception as e:
                logging.info(f"Fail in the calculations of metrics : {e}")
            

        else :
            st.info("Select an asset to view Metrics perfomance.")


#------------------------------------------------ TAB 4 : ESG STATISTICS  ---------------------------------------------------------------#
    with tab4:
        st.subheader("ESG Statistics")
        
        esg_data = pd.DataFrame()
        for ticker in sel_tickers_list:
            try:

                esg_data_ticker = fn.fetch_esg_data(ticker)
                if not esg_data_ticker.empty:
                    
                    esg_data = pd.concat([esg_data,esg_data_ticker])

                    esg_data = esg_data[['Ticker', 'totalEsg', 'environmentScore',	'socialScore',
                        'governanceScore',	'ratingYear',	'ratingMonth',	'highestControversy',
                        'peerCount',	'esgPerformance',	'peerGroup',	'relatedControversy',
                        'peerEsgScorePerformance',	'peerGovernancePerformance',	'peerSocialPerformance',
                        'peerEnvironmentPerformance',	'peerHighestControversyPerformance']]
                
                else:
                    st.error(f"INFO : The ticker {ticker} has empty ESG data")
                    logging.info(f"Aucune ESG Data pour {ticker}")
                    continue
                

            except Exception:
                pass

        esg_data = esg_data.rename(columns= {"environmentScore" : "Environment Score",
                                                "socialScore" : "Social Score", 
                                                "governanceScore" : "Governance Score",
                                                "totalEsg" : "ESG Score",
                                                "peerGroup": "Sector"}
                                                )


        esg_data_cleaned = fn.clean_esg_data(esg_data)


        st.write("ESG Cleaned data", esg_data_cleaned)

        esg_melted = esg_data_cleaned.melt(
            id_vars = ["Ticker"],
            value_vars = ["Environment Score", "Social Score", "Governance Score"],
            var_name = "Category",
            value_name = "Score ESG"
        )

        fig_esg = px.bar(
            esg_melted,
            x="Ticker",
            y="Score ESG",
            color="Category",
            barmode="group",
            title=f"ESG Scores by Ticker (in {esg_data_cleaned['ratingYear'].tolist()[0]})",
            labels={"Score": "Score", "Category": "ESG Category"},
        )
        st.plotly_chart(fig_esg, use_container_width=True)




        #Individual identity ESG card 
        st.subheader("Individual ESG Analysis")

        esg_valid_tickers = [ticker for ticker in valid_sel_tickers if not esg_data_cleaned[esg_data_cleaned['Ticker'] == ticker].empty]


        tab4_cols = st.columns(2)
        
        for i, ticker in enumerate(esg_valid_tickers):
            #Adding logo 

            if esg_data_cleaned.empty:
                    
                    st.error("Pas assez de données pour approfondir l'analyse")
                    logging.error(f"Pas de de donnée ESG pour approfondir l'analyse for {ticker}")
                    pass    
            else:
        
                try:

                    tab4_cols[i % 2].image(fn.fetch_logo(ticker), width = 65)
                    tab4_cols[i % 2].write(f"{str(esg_data_cleaned[esg_data_cleaned['Ticker'] == ticker ]['Sector'].iloc[0])}")


                    #Stock Metrics
                    tab4_cols2 = tab4_cols[i % 2].columns(3)
                    
                    #st.write(esg_data_cleaned[esg_data_cleaned['Ticker'] == ticker ])
                    tab4_cols2[0].metric(label = 'Environment Score', 
                                        value = esg_data_cleaned[esg_data_cleaned['Ticker'] == ticker ]['Environment Score'].round(2))
                    tab4_cols2[1].metric(label = 'Social Score', 
                                        value = esg_data_cleaned[esg_data_cleaned['Ticker'] == ticker ]['Social Score'].round(2))
                    tab4_cols2[2].metric(label = 'Gouvernance Score', 
                                        value = esg_data_cleaned[esg_data_cleaned['Ticker'] == ticker ]['Governance Score'].round(2))
                

                    #Graphs Sectors Performance Comparison
                    data_esg_graph = esg_data_cleaned.rename(columns={'peerEsgScorePerformance - Minimum Score' : 'Minimum Score',
                                                                    'peerEsgScorePerformance - Average Score' : 'Average Score',
                                                                    'peerEsgScorePerformance - Maximum Score' : 'Maximum Score'   
                                                                    })

                    data_esg_graph_melted = data_esg_graph.melt(id_vars = 'Ticker', 
                    value_vars = ['Minimum Score', 'Average Score', 'Maximum Score','ESG Score'],
                    var_name = 'Performance Type', 
                    value_name = 'Score'
                    )


                    fig_esg_perf = px.bar(data_esg_graph_melted[data_esg_graph_melted["Ticker"] == ticker], 
                                x = 'Ticker', 
                                y = 'Score', 
                                color = 'Performance Type', 
                                barmode = 'group', 
                                title = f'{ticker} sector ESG Performance',
                                labels = {'Score': 'Score', 'Performance Type': 'Performance'})

                    tab4_cols[i % 2].plotly_chart(fig_esg_perf,use_container_width=True)

                    tab4_cols[i % 2].write(f"STATUS : {esg_data_cleaned['ESG Status'].iloc[0]}")

                
                    
                except Exception as e:
                    st.error(f"Pas assez de données ESG pour le ticker {ticker}")
                    logging.error(f"Problème dans le calcul des données ESG : {e}")
    
                            
#---------------------------------------------- TAB 5 : INVEST WITHOUT CARBON  ---------------------------------------------------#

    with tab5:
      
    #    Les calculs et la méthodologie ici présents se basent sur un article scientifique :
    #    'ESG-Valued Portfolio Optimization and Dynamic Asset Pricing'
    #    Davide Lauria1,*, W. Brent Lindquist1, Stefan Mittnik2, and Svetlozar T. Rachev1
    #   https://arxiv.org/pdf/2206.02854  (page 8-9)

    
        st.subheader("Invest Without Carbon")

        # Vérifier si des tickers sont sélectionnés
        if len(sel_tickers_list) == 0:
            st.info("Please select at least one ticker to view the analysis.")
        else:
            financial_returns = yfdata[['Date', 'ticker', 'price_pct_daily']].pivot(index='Date', 
                                                                                    columns='ticker', 
                                                                                    values='price_pct_daily'
                                                                                    ).dropna()

            #Simulation des données ESG
            esg_scores = pd.DataFrame({ticker: np.random.uniform(50, 100, len(financial_returns)) for ticker in valid_sel_tickers}, 
                                      index=financial_returns.index)

            # Normalisation des scors ESG 
            esg_scores_normalized = 2 * (esg_scores - esg_scores.min()) / (esg_scores.max() - esg_scores.min()) - 1

            # Slider pour le paramètre lambda
            lambda_value = st.slider("Select ESG Affinity", 
                                     min_value = 0.0, 
                                     max_value = 1.0, 
                                     value = 0.5, 
                                     step = 0.1
                                     )

            # Adjusted returns 
            esg_adjusted_returns = lambda_value * (esg_scores_normalized / 255) + (1 - lambda_value) * financial_returns

            st.write("ESG Adjusted Returns", esg_adjusted_returns)

            st.subheader("ESG-Adjusted Returns")
            for ticker in valid_sel_tickers: #sel_tickers_list
                fig_esg_r = px.line(
                    esg_adjusted_returns, 
                    x = esg_adjusted_returns.index, 
                    y = ticker, 
                    title = f"{ticker} - ESG-Adjusted Returns",
                    markers = True,
                    labels = {"value": "Adjusted Return", "index": "Date"}
                )
                st.plotly_chart(fig_esg_r, use_container_width=True)

            # Efficient Frontier
            st.subheader("Efficient Frontier")
            mean_returns = esg_adjusted_returns.mean()
            risks = esg_adjusted_returns.std()
            fig_ef = px.scatter(
                x = risks, 
                y = mean_returns, 
                text = valid_sel_tickers, #sel_tickers_list
                title = "Efficient Frontier",
                labels = {"x": "Risk (Std Dev)", "y": "Return (Mean)"}
            )
            fig_ef.update_traces(textposition="top center")
            st.plotly_chart(fig_ef, use_container_width=True)



#-----------------------------------------------------------------------------------------------#

    if generate_btn:
        html_report = fn.generate_html_report(yfdata, 
                                              sel_dt1, sel_dt2, 
                                              esg_data_cleaned, 
                                              all_correlations_table)
        st.sidebar.download_button(
            label = "Download HTML Report",
            data = html_report,
            file_name = "portfolio_analysis_report.html",
            mime = "text/html"
        )



