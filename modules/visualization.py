import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def plot_carbon_emissions(carbon_data, df):
    """Create carbon emission visualization"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Monthly Carbon Emissions', 'Emissions by Category', 
                       'Carbon vs Spending', 'Sustainability Score'),
        specs=[[{"secondary_y": False}, {"type": "pie"}],
               [{"secondary_y": True}, {"type": "indicator"}]]
    )
    
    # Monthly emissions trend
    monthly_data = carbon_data['monthly_emissions'].reset_index()
    monthly_data['Month'] = monthly_data['Date'].astype(str)
    
    if not monthly_data.empty:
        fig.add_trace(
            go.Scatter(x=monthly_data['Month'], y=monthly_data['Carbon_Emissions_kg'],
                      mode='lines+markers', name='Monthly Emissions',
                      line=dict(color='red', width=3)),
            row=1, col=1
        )
    
    # Category emissions pie chart
    category_data = carbon_data['category_emissions']
    if not category_data.empty:
        fig.add_trace(
            go.Pie(labels=category_data.index, values=category_data.values,
                   hole=0.4, textinfo='label+percent',
                   marker_colors=['red' if 'High-Carbon' in label or label in ['Shopping', 'Food & Dining'] 
                                 else 'green' if 'Green' in label or 'Sustainable' in label 
                                 else 'orange' for label in category_data.index]),
            row=1, col=2
        )
    
    # Carbon vs Spending correlation
    monthly_spending = df.groupby(df['Date'].dt.to_period('M'))['Debit'].sum().reset_index()
    monthly_emissions_df = carbon_data['monthly_emissions'].reset_index()
    
    if not monthly_spending.empty and not monthly_emissions_df.empty:
        monthly_spending['Month'] = monthly_spending['Date'].astype(str)
        monthly_emissions_df['Month'] = monthly_emissions_df['Date'].astype(str)
        
        fig.add_trace(
            go.Bar(x=monthly_spending['Month'], y=monthly_spending['Debit'],
                   name='Monthly Spending', marker_color='lightblue', opacity=0.7),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=monthly_emissions_df['Month'], y=monthly_emissions_df['Carbon_Emissions_kg'],
                      mode='lines+markers', name='Carbon Emissions',
                      line=dict(color='red', width=2), yaxis='y2'),
            row=2, col=1, secondary_y=True
        )
    
    # Sustainability score gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=carbon_data['sustainability_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sustainability Score"},
            delta={'reference': 70},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 30], 'color': "red"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Carbon Emission Dashboard",
        title_x=0.5
    )
    
    # Update axes
    fig.update_xaxes(title_text="Month", row=1, col=1)
    fig.update_yaxes(title_text="CO₂ Emissions (kg)", row=1, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Spending (₹)", row=2, col=1)
    fig.update_yaxes(title_text="CO₂ Emissions (kg)", row=2, col=1, secondary_y=True)
    
    return fig

def create_main_dashboard(dashboard_data):
    """Create the main dashboard visualization"""
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Monthly Cash Flow', 'Category-wise Spending', 
                       'Income vs Expenses', 'Sustainable vs High-Carbon Spending',
                       'Monthly Carbon Footprint', 'Transaction Timeline'),
        specs=[[{"secondary_y": False}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "bar"}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    monthly_data = dashboard_data['monthly_summary']
    
    # Monthly Cash Flow
    fig.add_trace(
        go.Bar(x=monthly_data['Month'], y=monthly_data['Credit'], 
               name='Income', marker_color='green', opacity=0.7),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=monthly_data['Month'], y=-monthly_data['Debit'], 
               name='Expenses', marker_color='red', opacity=0.7),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=monthly_data['Month'], y=monthly_data['Net'],
                  mode='lines+markers', name='Net Savings', 
                  line=dict(color='blue', width=3)),
        row=1, col=1
    )
    
    # Category-wise Spending Pie
    category_data = dashboard_data['category_spending']
    if not category_data.empty:
        colors = ['green' if 'Green' in cat or 'Sustainable' in cat 
                 else 'red' if cat in ['High-Carbon Transport', 'Shopping', 'Food & Dining']
                 else 'orange' for cat in category_data.index]
        
        fig.add_trace(
            go.Pie(labels=category_data.index, values=category_data.values,
                   hole=0.3, textinfo='label+percent',
                   marker_colors=colors),
            row=1, col=2
        )
    
    # Income vs Expenses Bar
    fig.add_trace(
        go.Bar(x=['Total'], y=[dashboard_data['total_income']], 
               name='Total Income', marker_color='green'),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=['Total'], y=[dashboard_data['total_expenses']], 
               name='Total Expenses', marker_color='red'),
        row=2, col=1
    )
    
    # Sustainable vs High-Carbon Spending
    fig.add_trace(
        go.Bar(x=['Green Spending', 'High-Carbon Spending'], 
               y=[dashboard_data['green_spending'], dashboard_data['high_carbon_spending']],
               marker_color=['green', 'red'],
               name='Spending Comparison'),
        row=2, col=2
    )
    
    # Monthly Carbon Footprint
    fig.add_trace(
        go.Scatter(x=monthly_data['Month'], y=monthly_data['Carbon_Emissions_kg'],
                  mode='lines+markers', name='Carbon Emissions',
                  line=dict(color='darkred', width=3),
                  fill='tonexty'),
        row=3, col=1
    )
    
    # Transaction Timeline (Sample of recent transactions)
    df = dashboard_data['df']
    recent_transactions = df.nlargest(10, 'Date')
    
    fig.add_trace(
        go.Scatter(x=recent_transactions['Date'], 
                  y=recent_transactions['Debit'],
                  mode='markers',
                  marker=dict(size=8, color='red'),
                  name='Recent Expenses',
                  text=recent_transactions['Particulars'],
                  textposition="top center"),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1000,
        showlegend=True,
        title_text="Comprehensive Sustainable Finance Dashboard",
        title_x=0.5
    )
    
    return fig