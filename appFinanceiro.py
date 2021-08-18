# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:48:46 2021

@author: Pamela
"""
# Imports

import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from datetime import date
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import warnings
warnings.filterwarnings("ignore")

data_inicio = '2016-01-01'
data_hoje = date.today().strftime("%Y-%m-%d")

st.title("Dashboard Financeiro Interativo e em Tempo Real Para Previsão de Ativos Financeiros")
empresas = ('PFE', 'MRNA', 'BAC', 'AAPL' )
selecao_empresa = st.selectbox('Selecione a Empresa para as Previsões: ', empresas)

@st.cache

def carregar_dados(acao):
    dados = yf.download(acao, data_inicio, data_hoje)
    dados.reset_index(inplace = True)
    return dados

mensagem = st.text('Carregando Dados...')

dados = carregar_dados(selecao_empresa)

mensagem.text("Carga de Dados Concluída!!!")

st.subheader("Visualização dos Dados Brutos")
st.write(dados.tail(10))

def grafico_dados_brutos():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = dados['Date'], y = dados['Open'], name = "stock_open"))
    fig.add_trace(go.Scatter(x = dados['Date'], y = dados['Close'], name = "stock_close"))
    fig.layout.update(title_text = 'Preço de Abertura e Fechamento', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

grafico_dados_brutos()

st.subheader('Previsões com Machine Learning')

treino = dados[['Date', 'Close']]
treino = treino.rename(columns = {"Date": "ds", "Close": "y"})

#Criando o Modelo
modelo = Prophet()

modelo.fit(treino)

num_anos = st.slider('Horizonte de Previsão (em anos):', 1, 4)
periodo = num_anos*365
futuro = modelo.make_future_dataframe(periods = periodo)

previsoes = modelo.predict(futuro)

st.subheader('Dados Previstos:')
st.write(previsoes.tail(10))

st.subheader('Previsão de Preço dos /tivos Financeiros para o Período Selecionado')

grafico2 = plot_plotly(modelo, previsoes)
st.plotly_chart(grafico2)







