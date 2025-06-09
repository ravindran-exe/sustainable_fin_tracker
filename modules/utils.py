import json
import pandas as pd
import streamlit as st

def safe_to_json(df, orient='records', date_format='iso'):
    """Safely convert a DataFrame to JSON, handling recursive structures."""
    try:
        # Flatten any recursive or nested structures
        df = df.applymap(lambda x: str(x) if isinstance(x, (dict, list, set)) else x)
        
        # Convert to JSON
        json_data = df.to_json(orient=orient, date_format=date_format)
        return json_data if json_data else "{}"  # Ensure non-None return
    except Exception as e:
        st.error(f"Error converting DataFrame to JSON: {e}")
        return "{}"  # Return empty JSON string on error