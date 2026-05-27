import streamlit as st

home = st.Page("pages/home.py", title="Home", icon=":material/home:")
predict = st.Page("pages/predict.py", title="Demo", icon=":material/search:")

pg = st.navigation([home, predict], position="hidden")
pg.run()
