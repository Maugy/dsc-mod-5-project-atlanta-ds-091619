#SessionState Demo

import streamlit as st
import SessionState
import terms

search_terms = terms.total_terms
# time_series = pd.read_csv('time_series.csv')

skill = st.sidebar.selectbox('Skill Comparison: ', options=search_terms, index=3)

st.sidebar.title('CLIMATE FORECAST')

# model_choice = st.sidebar.selectbox('MODELS:', ['CFSv2', 'GFDL'])
# var_choice = st.sidebar.selectbox('VARS:', ['PCP', 'TEMP'])

fig = f'{skill}'

img = SessionState.get(img = None)

submit = st.sidebar.button('submit')
if submit:
	img.img = open(fig, 'rb').read()

try:
	st.image(img.img)
except:
	pass