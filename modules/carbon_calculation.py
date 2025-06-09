import pandas as pd
import streamlit as st

def calculate_carbon_emissions(df):
    """Calculate carbon emissions for each transaction"""
    
    # Carbon emission factors (kg CO2 per rupee spent)
    carbon_factors = {
        'Sustainable Transport': 0.02,
        'High-Carbon Transport': 0.15,
        'Green Energy': 0.0,
        'Utilities': 0.08,
        'Sustainable Food': 0.03,
        'Food & Dining': 0.12,
        'Green Shopping': 0.02,
        'Shopping': 0.10,
        'Green Investment': -0.05,  # Carbon negative
        'Healthcare': 0.06,
        'Housing': 0.09,
        'Digital Services': 0.01,
        'Others': 0.07
    }
    
    df['Carbon_Emissions_kg'] = 0.0
    
    for idx, row in df.iterrows():
        category = row['Category']
        amount = row['Debit'] if row['Debit'] > 0 else 0  # Only expenses contribute to emissions
        
        if category in carbon_factors:
            emissions = amount * carbon_factors[category]
            df.loc[idx, 'Carbon_Emissions_kg'] = max(0, emissions)  # Ensure non-negative
    
    return df

def create_carbon_dashboard(df):
    """Create carbon emission analysis dashboard"""
    
    # Calculate total emissions
    total_emissions = df['Carbon_Emissions_kg'].sum()
    monthly_emissions = df.groupby(df['Date'].dt.to_period('M'))['Carbon_Emissions_kg'].sum()
    category_emissions = df.groupby('Category')['Carbon_Emissions_kg'].sum().sort_values(ascending=False)
    
    # Calculate sustainability score (0-100)
    high_carbon_spending = df[df['Category'].isin(['High-Carbon Transport', 'Shopping', 'Food & Dining'])]['Debit'].sum()
    green_spending = df[df['Category'].str.contains('Green|Sustainable', na=False)]['Debit'].sum()
    total_spending = df['Debit'].sum()
    
    if total_spending > 0:
        sustainability_score = max(0, min(100, 
            100 - (high_carbon_spending / total_spending * 50) + (green_spending / total_spending * 30)
        ))
    else:
        sustainability_score = 50
    
    # Carbon offset needed (in trees)
    trees_needed = total_emissions / 22  # One tree absorbs ~22kg CO2 per year
    
    return {
        'total_emissions': total_emissions,
        'monthly_emissions': monthly_emissions,
        'category_emissions': category_emissions,
        'sustainability_score': sustainability_score,
        'trees_needed': trees_needed,
        'green_spending': green_spending,
        'high_carbon_spending': high_carbon_spending
    }