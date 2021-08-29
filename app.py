import streamlit as st
import datetime
import pandas as pd
import cufflinks as cf
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# App title
st.markdown('''
# Stock price prediction app
shon are the stock price data for query companies

**Credits**
- App built by Rahul Gupta
- Built in `Python` using `streamlit`,`yfinance`, `cufflinks`, `pandas` and `datetime`
''')

st.write('---')

# Sidebar
st.sidebar.subheader('Query parameters')
start_date = st.sidebar.date_input("Start date",datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End date",datetime.date(2021, 1, 31))

#retrieving ticker data
ticker_list = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt")
ticker_symbol = st.sidebar.selectbox('Stock ticker',ticker_list)
ticker_data = yf.Ticker(ticker_symbol)
tickerdf = ticker_data.history(period='1d',start=start_date,end=end_date)
tickerdf.reset_index(inplace=True)

#ticker info
string_logo = '<img src=%s>' % ticker_data.info['logo_url']
st.markdown(string_logo,unsafe_allow_html=True)

#st.write(ticker_data.info)

string_name = ticker_data.info['longName']
st.header('**%s**' % string_name)

string_sector = ticker_data.info['sector']
st.header(string_sector)

string_summary = ticker_data.info['longBusinessSummary']
st.write(string_summary)

#ticker data
st.header('**Ticker data**')
st.write(tickerdf)

# Bollinger bands
st.header('**Bollinger Bands**')
qf=cf.QuantFig(tickerdf,title='First Quant Figure',legend='top',name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)

#forcasting

n_years = st.slider("Year of Prediction",1,4)
period = n_years * 365

df_train = tickerdf[['Date','Close']]
df_train = df_train.rename(columns={"Date":"ds","Close":"y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forcast = m.predict(future)

st.subheader('Forcast data')
st.write(forcast.tail())

st.write('forcast data')
fig1 = plot_plotly(m,forcast)
st.plotly_chart(fig1)

st.write('forcast components')
fig2 = m.plot_components(forcast)
st.write(fig2)


####
#st.write('---')
#st.write(tickerData.info)
