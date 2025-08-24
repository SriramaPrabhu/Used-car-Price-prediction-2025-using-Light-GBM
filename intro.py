import streamlit as st

st.title("My first PoC in Streamlit :blue[cool] :sunglasses:")

import streamlit as st

agree = st.checkbox("I agree")

if agree:
    st.write("Great!")

import time
import streamlit as st

with st.status("Downloading data..."):
    st.write("Searching for data...")
    time.sleep(2)
    st.write("Found URL.")
    time.sleep(1)
    st.write("Downloading data...")
    time.sleep(1)

st.button("Rerun")