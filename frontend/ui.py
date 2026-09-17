import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/ask"


st.title("ResolveAI")

query = st.text_input(
    "Customer message",
    placeholder="Describe your Spotify issue..."
)

if st.button("Ask") and query.strip():

    try:
        response = requests.post(
            API_URL,
            params={"query": query},
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        with st.expander("View detailed result", expanded=False):
            st.json(data)

    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")