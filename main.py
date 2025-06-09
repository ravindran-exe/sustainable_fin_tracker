import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
import tempfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import io
import base64

# Configure Gemini API key
genai.configure(api_key=st.secrets.get("GOOGLE_API_KEY", ""))

# --- Enhanced Text Extraction ---
def extract_text_from_pdf(file):
    """Enhanced PDF text extraction with better formatting preservation"""
    text = ""
    tables_data = []
    
    with pdfplumber.open(file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Extract text
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text + "\n"
            
            # Extract tables
            tables = page.extract_tables()
            if tables:
                for table_num, table in enumerate(tables):
                    tables_data.append({
                        'page': page_num + 1,
                        'table': table_num + 1,
                        'data': table
                    })
    
    return text, tables_data

def extract_text_with_ocr(file):
    """Enhanced OCR extraction with preprocessing"""
    images = []
    full_text = ""
    
    with pdfplumber.open(file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Convert to image with higher resolution
            img = page.to_image(resolution=400)
            
            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                img.save(tmp_img.name, format="PNG")
                pil_img = Image.open(tmp_img.name)
                
                # OCR with better configuration
                custom_config = r'--oem 3 --psm 6'
                ocr_text = pytesseract.image_to_string(pil_img, config=custom_config)
                
                full_text += f"\n--- Page {page_num + 1} (OCR) ---\n"
                full_text += ocr_text + "\n"
    
    return full_text

# --- Intelligent Chunking ---
def smart_chunk_text(text, max_chunk_size=1500, overlap=200):
    """Smart chunking that preserves context"""
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Add overlap
                words = current_chunk.split()
                if len(words) > overlap // 10:
                    current_chunk = " ".join(words[-(overlap // 10):]) + " " + sentence + ". "
                else:
                    current_chunk = sentence + ". "
            else:
                current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# --- Enhanced Embedding ---
def embed_texts_batch(texts, batch_size=10):
    """Batch embedding for efficiency"""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = []
        
        for text in batch:
            try:
                response = genai.embed_content(
                    model="models/embedding-001",
                    content=text
                )
                batch_embeddings.append(response['embedding'])
            except Exception as e:
                st.warning(f"Embedding failed for a chunk: {str(e)}")
                # Use zero vector as fallback
                batch_embeddings.append([0.0] * 768)
        
        embeddings.extend(batch_embeddings)
    
    return embeddings

# --- Enhanced Vector Search ---
def retrieve_relevant_chunks(query, chunks, chunk_embeddings, top_k=5):
    """Enhanced retrieval with better ranking"""
    try:
        query_response = genai.embed_content(
            model="models/embedding-001",
            content=query
        )
        query_emb = query_response['embedding']
        
        # Calculate similarities
        similarities = cosine_similarity([query_emb], chunk_embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        relevant_chunks = []
        scores = []
        
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for relevance
                relevant_chunks.append(chunks[idx])
                scores.append(similarities[idx])
        
        return relevant_chunks, scores
    
    except Exception as e:
        st.error(f"Retrieval failed: {str(e)}")
        return [], []

# --- FIXED Advanced Transaction Parsing ---
def parse_transactions_advanced(text, tables_data=None):
    """FIXED: Advanced transaction parsing with better JSON handling and validation"""
    try:
        # Strategy 1: Use AI model to parse transactions
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        json_text = re.sub(r'```json\s*', '', json_text)
        json_text = re.sub(r'```\s*', '', json_text)
        json_text = json_text.strip()

        start_idx = json_text.find('[')
        end_idx = json_text.rfind(']')

        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            st.warning("No valid JSON array found in AI response, trying regex parsing...")
            return regex_parse_transactions(text)

        json_data_str = json_text[start_idx:end_idx + 1]

        try:
            json_data = json.loads(json_data_str)
        except json.JSONDecodeError as e:
            st.warning(f"JSON parsing failed: {str(e)}, trying to fix common issues...")
            json_data_str = fix_json_string(json_data_str)
            json_data = json.loads(json_data_str)

        if not isinstance(json_data, list) or len(json_data) == 0:
            st.warning("Invalid JSON structure, falling back to regex parsing...")
            return regex_parse_transactions(text)

        df = pd.DataFrame(json_data)

        column_mapping = {
            'description': 'Particulars',
            'particulars': 'Particulars',
            'details': 'Particulars',
            'narration': 'Particulars',
            'debit': 'Debit',
            'credit': 'Credit',
            'balance': 'Balance',
            'date': 'Date'
        }

        df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)

        required_cols = ['Date', 'Particulars', 'Debit', 'Credit', 'Balance']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0 if col in ['Debit', 'Credit', 'Balance'] else 'N/A'

        df = clean_transaction_data(df)

        if len(df) > 0:
            st.success(f"✅ Successfully parsed {len(df)} transactions using AI")
            return df
        else:
            st.warning("No valid transactions found, trying regex parsing...")
            return regex_parse_transactions(text)

    except Exception as e:
        st.warning(f"AI parsing failed: {str(e)}, trying regex parsing...")
        return regex_parse_transactions(text)

def fix_json_string(json_str):
    """Fix common JSON formatting issues"""
    # Remove trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Fix unquoted field names
    json_str = re.sub(r'(\w+):', r'"\1":', json_str)
    
    # Fix single quotes to double quotes
    json_str = json_str.replace("'", '"')
    
    return json_str

def clean_transaction_data(df):
    """FIXED: Clean and validate transaction data with better number parsing"""
    try:
        # Clean Date column
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        
        # Clean numeric columns with better handling
        for col in ['Debit', 'Credit', 'Balance']:
            if col in df.columns:
                # Convert to string first, handle various formats
                df[col] = df[col].astype(str)
                
                # Remove common non-numeric characters
                df[col] = df[col].str.replace(',', '', regex=False)
                df[col] = df[col].str.replace('₹', '', regex=False)
                df[col] = df[col].str.replace('Rs', '', regex=False)
                df[col] = df[col].str.replace('Rs.', '', regex=False)
                df[col] = df[col].str.replace('INR', '', regex=False)
                df[col] = df[col].str.replace('(', '-', regex=False)  # Handle negative in parentheses
                df[col] = df[col].str.replace(')', '', regex=False)
                df[col] = df[col].str.strip()
                
                # Replace empty strings and 'N/A' with 0
                df[col] = df[col].replace(['', 'N/A', 'n/a', 'null', 'None'], '0')
                
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                # Ensure non-negative values for debit/credit (handle data entry errors)
                df[col] = df[col].abs()
        
        # Clean Particulars column
        if 'Particulars' in df.columns:
            df['Particulars'] = df['Particulars'].astype(str).str.strip()
            df['Particulars'] = df['Particulars'].replace(['nan', 'None', 'null'], 'N/A')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['Date'])
        
        # Additional validation: Remove rows where all amounts are 0
        df = df[~((df['Debit'] == 0) & (df['Credit'] == 0) & (df['Balance'] == 0))]
        
        # FIXED: Validate transaction logic - if both debit and credit are > 0, it's likely an error
        mask = (df['Debit'] > 0) & (df['Credit'] > 0)
        if mask.sum() > 0:
            # For rows with both debit and credit, keep the larger amount and zero the smaller
            for idx in df[mask].index:
                if df.loc[idx, 'Debit'] > df.loc[idx, 'Credit']:
                    df.loc[idx, 'Credit'] = 0
                else:
                    df.loc[idx, 'Debit'] = 0
        
        return df
        
    except Exception as e:
        st.error(f"Error cleaning transaction data: {str(e)}")
        return df

def regex_parse_transactions(text):
    """IMPROVED: Fallback regex-based transaction parsing with better patterns"""
    transactions = []
    
    # More comprehensive patterns for different bank statement formats
    patterns = [
        # Pattern 1: DD/MM/YYYY Description Amount Amount Amount
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+([^0-9\n]+?)\s+([0-9,]+\.?\d*)\s+([0-9,]+\.?\d*)\s+([0-9,]+\.?\d*)',
        
        # Pattern 2: DD-MM-YYYY Description Debit Credit Balance
        r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s+([A-Za-z][^\d\n]*?)\s+(?:(\d+[,\d]*\.?\d*)|(?:\s+))\s+(?:(\d+[,\d]*\.?\d*)|(?:\s+))\s+(\d+[,\d]*\.?\d*)',
        
        # Pattern 3: Date followed by description and one or more amounts
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+([A-Za-z].*?)\s+(\d+[,\d]*\.?\d*)(?:\s+(\d+[,\d]*\.?\d*))?(?:\s+(\d+[,\d]*\.?\d*))?',
        
        # Pattern 4: Handle formats with tabs or multiple spaces
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(.+?)\s+(\d+(?:,\d{3})*(?:\.\d{2})?)',
    ]
    
    for pattern_idx, pattern in enumerate(patterns):
        matches = re.findall(pattern, text, re.MULTILINE)
        
        if matches:
            st.info(f"Using regex pattern {pattern_idx + 1}, found {len(matches)} potential transactions")
            
            for match in matches:
                try:
                    date_str = match[0]
                    particulars = match[1].strip() if len(match) > 1 else 'N/A'
                    
                    # Extract amounts from the remaining groups
                    amounts = []
                    for i in range(2, len(match)):
                        if match[i] and match[i].strip():
                            # Clean amount string
                            amount_str = match[i].replace(',', '').strip()
                            if amount_str and amount_str.replace('.', '').isdigit():
                                amounts.append(float(amount_str))
                    
                    if not amounts:
                        continue
                    
                    # Determine debit, credit, balance based on number of amounts
                    if len(amounts) == 1:
                        # Single amount - could be debit or credit, assume debit if positive
                        debit = amounts[0] if amounts[0] > 0 else 0
                        credit = 0
                        balance = 0
                    elif len(amounts) == 2:
                        # Two amounts - likely debit and balance OR credit and balance
                        debit = amounts[0]
                        credit = 0
                        balance = amounts[1]
                    elif len(amounts) >= 3:
                        # Three amounts - debit, credit, balance
                        debit = amounts[0]
                        credit = amounts[1]
                        balance = amounts[2]
                    else:
                        continue
                    
                    # Basic validation
                    if len(particulars) < 3:  # Skip very short descriptions
                        continue
                        
                    transactions.append({
                        'Date': date_str,
                        'Particulars': particulars,
                        'Debit': debit,
                        'Credit': credit,
                        'Balance': balance
                    })
                    
                except Exception as e:
                    continue
            
            # If we found transactions with this pattern, use them
            if transactions:
                break
    
    if transactions:
        df = pd.DataFrame(transactions)
        df = clean_transaction_data(df)
        st.success(f"✅ Successfully parsed {len(df)} transactions using regex")
        return df
    
    st.error("❌ Unable to parse transactions with any method")
    return pd.DataFrame()

# --- SUSTAINABLE FINANCE CATEGORIZATION ---
def smart_categorize_transaction(particulars, amount=0, is_credit=True):
    """Enhanced categorization for sustainable finance tracking"""
    if not particulars or pd.isna(particulars):
        return 'Others'
    
    p = str(particulars).lower().strip()
    
    if is_credit:  # Income categories
        income_categories = {
            'Green Income': {
                'keywords': ['solar', 'renewable', 'green bond', 'sustainable', 'esg dividend', 'carbon credit'],
                'weight': 1.0
            },
            'Salary': {
                'keywords': ['salary', 'sal cr', 'sal credit', 'wages', 'pay credit', 'payroll'],
                'weight': 0.9
            },
            'Investment Returns': {
                'keywords': ['dividend', 'interest credit', 'int cr', 'mutual fund', 'sip'],
                'weight': 0.8
            },
            'Other Income': {
                'keywords': ['refund', 'cashback', 'reward', 'bonus'],
                'weight': 0.7
            }
        }
        
        best_category = 'Other Income'
        best_score = 0
        
        for category, data in income_categories.items():
            score = sum(data['weight'] for keyword in data['keywords'] if keyword in p)
            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category
    
    else:  # Expense categories with sustainability focus
        expense_categories = {
            'Sustainable Transport': {
                'keywords': ['metro', 'bus', 'train', 'electric', 'ev charging', 'bike share', 'cycle'],
                'weight': 1.0,
                'carbon_factor': 0.05  # Low carbon
            },
            'High-Carbon Transport': {
                'keywords': ['fuel', 'petrol', 'diesel', 'gas station', 'taxi', 'uber', 'ola', 'flight'],
                'weight': 0.9,
                'carbon_factor': 2.3  # High carbon
            },
            'Green Energy': {
                'keywords': ['solar panel', 'renewable energy', 'green electricity', 'wind power'],
                'weight': 1.0,
                'carbon_factor': 0.0  # Zero carbon
            },
            'Utilities': {
                'keywords': ['electricity', 'water', 'gas bill', 'utility'],
                'weight': 0.8,
                'carbon_factor': 0.5  # Medium carbon
            },
            'Sustainable Food': {
                'keywords': ['organic', 'local farm', 'vegan', 'plant based', 'farmers market'],
                'weight': 1.0,
                'carbon_factor': 0.3  # Low carbon
            },
            'Food & Dining': {
                'keywords': ['restaurant', 'food', 'dining', 'swiggy', 'zomato', 'grocery', 'supermarket'],
                'weight': 0.7,
                'carbon_factor': 1.2  # Medium-high carbon
            },
            'Green Shopping': {
                'keywords': ['eco friendly', 'sustainable', 'organic store', 'green product'],
                'weight': 1.0,
                'carbon_factor': 0.2  # Low carbon
            },
            'Shopping': {
                'keywords': ['amazon', 'flipkart', 'shopping', 'mall', 'fashion'],
                'weight': 0.6,
                'carbon_factor': 1.5  # High carbon (fast fashion, shipping)
            },
            'Green Investment': {
                'keywords': ['esg fund', 'green bond', 'sustainable fund', 'solar investment'],
                'weight': 1.0,
                'carbon_factor': -0.5  # Negative carbon (offsetting)
            },
            'Healthcare': {
                'keywords': ['hospital', 'medical', 'pharmacy', 'doctor'],
                'weight': 0.8,
                'carbon_factor': 0.4
            },
            'Housing': {
                'keywords': ['rent', 'mortgage', 'emi', 'maintenance'],
                'weight': 0.7,
                'carbon_factor': 0.8
            },
            'Digital Services': {
                'keywords': ['internet', 'streaming', 'netflix', 'subscription', 'software'],
                'weight': 0.5,
                'carbon_factor': 0.1  # Low carbon
            }
        }
        
        best_category = 'Others'
        best_score = 0
        
        for category, data in expense_categories.items():
            score = sum(data['weight'] for keyword in data['keywords'] if keyword in p)
            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category

# --- CARBON EMISSION CALCULATION ---
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

# --- COMPREHENSIVE DASHBOARD ---
def create_comprehensive_dashboard(df):
    """Create a comprehensive sustainable finance dashboard"""
    
    # Add categories
    df['Category'] = df.apply(
        lambda row: smart_categorize_transaction(
            row['Particulars'], 
            row['Debit'] if row['Debit'] > 0 else row['Credit'],
            is_credit=(row['Credit'] > 0)
        ), 
        axis=1
    )
    
    # Calculate carbon emissions
    df = calculate_carbon_emissions(df)
    
    # Calculate metrics
    total_expenses = df['Debit'].sum()
    total_income = df['Credit'].sum()
    net_savings = total_income - total_expenses
    
    # Sustainable spending analysis
    green_spending = df[df['Category'].str.contains('Green|Sustainable', na=False)]['Debit'].sum()
    high_carbon_spending = df[df['Category'].isin(['High-Carbon Transport', 'Shopping', 'Food & Dining'])]['Debit'].sum()
    
    # Monthly analysis
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_summary = df.groupby('Month').agg({
        'Debit': 'sum',
        'Credit': 'sum',
        'Carbon_Emissions_kg': 'sum'
    }).reset_index()
    monthly_summary['Net'] = monthly_summary['Credit'] - monthly_summary['Debit']
    monthly_summary['Month'] = monthly_summary['Month'].astype(str)

    # Category-wise spending
    category_spending = df.groupby('Category')['Debit'].sum().sort_values(ascending=False)
    
    # Carbon analysis
    carbon_data = create_carbon_dashboard(df)
    
    return {
        'df': df,
        'total_expenses': total_expenses,
        'total_income': total_income,
        'net_savings': net_savings,
        'green_spending': green_spending,
        'high_carbon_spending': high_carbon_spending,
        'monthly_summary': monthly_summary,
        'category_spending': category_spending,
        'carbon_data': carbon_data
    }

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

# --- STREAMLIT APP ---
def main():
    st.set_page_config(
        page_title="🌱 Sustainable Finance Analyzer",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
    }
    .sustainability-score {
        font-size: 2rem;
        font-weight: bold;
        color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🌱 Sustainable Finance Analyzer</h1>
        <p>Transform your bank statements into actionable sustainability insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Analysis Options")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Bank Statement (PDF)",
            type=['pdf'],
            help="Upload your bank statement PDF for analysis"
        )
        
        # Analysis options
        st.subheader("🔧 Processing Options")
        use_ocr = st.checkbox("Use OCR for scanned PDFs", value=False)
        chunk_size = st.slider("Text Chunk Size", 1000, 3000, 1500)
        
        # Gemini API status
        api_key = st.secrets.get("GOOGLE_API_KEY", "")
        if api_key:
            st.success("✅ Gemini API Connected")
        else:
            st.error("❌ Gemini API Key Missing")
            st.info("Add GOOGLE_API_KEY to your Streamlit secrets")
    
    # Main content area
    if uploaded_file is not None:
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Transaction Analysis", 
            "🌍 Carbon Dashboard", 
            "📈 Financial Insights",
            "📊 Raw Data"
        ])
        
        # Process the file
        with st.spinner("🔄 Processing your bank statement..."):
            try:
                # Extract text
                if use_ocr:
                    text = extract_text_with_ocr(uploaded_file)
                    tables_data = []
                else:
                    text, tables_data = extract_text_from_pdf(uploaded_file)
                
                # Parse transactions
                df = parse_transactions_advanced(text, tables_data)
                
                if df.empty:
                    st.error("❌ No transactions found. Please check your PDF format or try enabling OCR.")
                    st.stop()
                
                # Create comprehensive dashboard
                dashboard_data = create_comprehensive_dashboard(df)
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.stop()
        
        # Tab 1: Transaction Analysis
        with tab1:
            st.header("📋 Transaction Analysis")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Income",
                    f"₹{dashboard_data['total_income']:,.2f}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Total Expenses", 
                    f"₹{dashboard_data['total_expenses']:,.2f}",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Net Savings",
                    f"₹{dashboard_data['net_savings']:,.2f}",
                    delta=f"₹{dashboard_data['net_savings']:,.2f}"
                )
            
            with col4:
                savings_rate = (dashboard_data['net_savings'] / dashboard_data['total_income'] * 100) if dashboard_data['total_income'] > 0 else 0
                st.metric(
                    "Savings Rate",
                    f"{savings_rate:.1f}%",
                    delta=f"{savings_rate:.1f}%"
                )
            
            # Main dashboard
            st.plotly_chart(
                create_main_dashboard(dashboard_data), 
                use_container_width=True
            )
            
            # Monthly summary table
            st.subheader("📅 Monthly Summary")
            st.dataframe(
                dashboard_data['monthly_summary'],
                use_container_width=True
            )
        
        # Tab 2: Carbon Dashboard
        with tab2:
            st.header("🌍 Carbon Footprint Analysis")
            
            carbon_data = dashboard_data['carbon_data']
            
            # Carbon metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total CO₂ Emissions",
                    f"{carbon_data['total_emissions']:.1f} kg",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Trees Needed",
                    f"{carbon_data['trees_needed']:.0f}",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Sustainability Score",
                    f"{carbon_data['sustainability_score']:.0f}/100",
                    delta=None
                )
            
            with col4:
                st.metric(
                    "Green Spending",
                    f"₹{carbon_data['green_spending']:,.0f}",
                    delta=None
                )
            
            # Carbon dashboard
            st.plotly_chart(
                plot_carbon_emissions(carbon_data, dashboard_data['df']),
                use_container_width=True
            )
            
            # Recommendations
            st.subheader("🌱 Sustainability Recommendations")
            
            if carbon_data['sustainability_score'] < 30:
                st.error("🚨 High Carbon Footprint - Immediate Action Needed")
                st.markdown("""
                - **Reduce high-carbon transport**: Consider public transport, cycling, or electric vehicles
                - **Shift to sustainable shopping**: Buy local, organic, and eco-friendly products
                - **Invest in green energy**: Solar panels, renewable energy plans
                """)
            elif carbon_data['sustainability_score'] < 70:
                st.warning("⚠️ Moderate Carbon Footprint - Room for Improvement")
                st.markdown("""
                - **Increase green investments**: ESG funds, sustainable bonds
                - **Optimize energy usage**: LED lights, energy-efficient appliances
                - **Choose sustainable food options**: Reduce meat consumption, buy local
                """)
            else:
                st.success("✅ Excellent Sustainability Score - Keep it up!")
                st.markdown("""
                - **Maintain green practices**: Continue sustainable choices
                - **Share your knowledge**: Inspire others to adopt sustainable practices
                - **Explore carbon offsetting**: Plant trees, support renewable projects
                """)
        
        # Tab 3: Financial Insights
        with tab3:
            st.header("📈 Financial Insights")
            
            # Spending patterns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💰 Top Spending Categories")
                category_chart = px.bar(
                    x=dashboard_data['category_spending'].values,
                    y=dashboard_data['category_spending'].index,
                    orientation='h',
                    title="Category-wise Spending"
                )
                st.plotly_chart(category_chart, use_container_width=True)
            
            with col2:
                st.subheader("📊 Monthly Trends")
                monthly_chart = px.line(
                    dashboard_data['monthly_summary'],
                    x='Month',
                    y=['Credit', 'Debit'],
                    title="Income vs Expenses Trend"
                )
                st.plotly_chart(monthly_chart, use_container_width=True)
            
            # Financial health indicators
            st.subheader("💡 Financial Health Indicators")
            
            # Calculate financial ratios
            expense_ratio = (dashboard_data['total_expenses'] / dashboard_data['total_income']) if dashboard_data['total_income'] > 0 else 0
            green_ratio = (dashboard_data['green_spending'] / dashboard_data['total_expenses']) if dashboard_data['total_expenses'] > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Expense Ratio", f"{expense_ratio:.1%}")
                if expense_ratio > 0.8:
                    st.error("High spending - consider budgeting")
                elif expense_ratio > 0.6:
                    st.warning("Moderate spending - room for savings")
                else:
                    st.success("Good spending control")
            
            with col2:
                st.metric("Green Spending Ratio", f"{green_ratio:.1%}")
                if green_ratio > 0.2:
                    st.success("Excellent sustainable choices")
                elif green_ratio > 0.1:
                    st.info("Good sustainable practices")
                else:
                    st.warning("Consider more green options")
            
            with col3:
                avg_transaction = dashboard_data['total_expenses'] / len(dashboard_data['df']) if len(dashboard_data['df']) > 0 else 0
                st.metric("Avg Transaction", f"₹{avg_transaction:.0f}")
        
        # Tab 4: Raw Data
        with tab4:
            st.header("📊 Raw Transaction Data")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                categories = ['All'] + list(dashboard_data['df']['Category'].unique())
                selected_category = st.selectbox("Filter by Category", categories)
            
            with col2:
                min_amount = st.number_input("Minimum Amount", value=0)
            
            with col3:
                date_range = st.date_input(
                    "Date Range",
                    value=(dashboard_data['df']['Date'].min(), dashboard_data['df']['Date'].max()),
                    min_value=dashboard_data['df']['Date'].min(),
                    max_value=dashboard_data['df']['Date'].max()
                )
            
            # Apply filters
            filtered_df = dashboard_data['df'].copy()
            
            if selected_category != 'All':
                filtered_df = filtered_df[filtered_df['Category'] == selected_category]
            
            if min_amount > 0:
                filtered_df = filtered_df[
                    (filtered_df['Debit'] >= min_amount) | 
                    (filtered_df['Credit'] >= min_amount)
                ]
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[
                    (filtered_df['Date'] >= pd.to_datetime(start_date)) &
                    (filtered_df['Date'] <= pd.to_datetime(end_date))
                ]
            
            # Display data
            st.dataframe(
                filtered_df[['Date', 'Particulars', 'Debit', 'Credit', 'Balance', 'Category', 'Carbon_Emissions_kg']],
                use_container_width=True
            )
            
            # Download options
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv_data,
                    "transactions.csv",
                    "text/csv"
                )
            
            with col2:
                json_data = safe_to_json(filtered_df, orient='records', date_format='iso')
                st.download_button(
                    "📥 Download JSON",
                    json_data,
                    "transactions.json",
                    "application/json"
                )
    
    else:
        # Landing page
        st.markdown("""
        ## 🌟 Welcome to Sustainable Finance Analyzer
        
        Transform your banking data into actionable sustainability insights with our AI-powered analyzer.
        
        ### ✨ Key Features:
        
        **📊 Smart Transaction Analysis**
        - Automatic categorization of expenses
        - Income vs expense tracking
        - Monthly spending patterns
        
        **🌍 Carbon Footprint Tracking**
        - Calculate CO₂ emissions from spending
        - Sustainability score calculation
        - Green vs high-carbon spending analysis
        
        **📈 Comprehensive Dashboards**
        - Visual spending analytics
        - Carbon emission trends
        - Financial health indicators
        
        ### 🚀 Getting Started:
        1. Upload your bank statement PDF using the sidebar
        2. Choose processing options (OCR for scanned documents)
        3. Explore your financial and environmental impact
        4. Get AI-powered recommendations for improvement
        
        ### 🔒 Privacy & Security:
        - Your data is processed locally
        - No financial information is stored
        - Secure AI analysis through Google's Gemini API
        """)
        
        # Sample metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("📊 **Transaction Analysis**\nSmart categorization and trend analysis")
        
        with col2:
            st.success("🌱 **Sustainability Tracking**\nCarbon footprint and green spending")
        
        with col3:
            st.warning("🤖 **AI Insights**\nPersonalized recommendations")

# Define the model and prompt variables
model = genai.GenerativeModel("gemini-1.5-flash")

prompt = """
You are an expert at extracting bank transaction data. Analyze the following bank statement text and extract ALL transactions.

CRITICAL INSTRUCTIONS:
1. Return ONLY a valid JSON array with no additional text, explanations, or markdown formatting
2. Each transaction must have these exact fields: "date", "description", "debit", "credit", "balance"
3. Use ONLY these field names (lowercase)
4. If debit/credit/balance is empty or not applicable, use 0 (zero)
5. If description is empty, use "N/A"
6. Format dates as YYYY-MM-DD
7. Remove all commas from numbers
8. Do not include any text before or after the JSON array

Text to analyze (first 4000 chars):
{text[:4000]}

JSON Array:
"""

def safe_to_json(df, orient='records', date_format='iso'):
    """Safely convert a DataFrame to JSON, handling recursive structures."""
    try:
        # Inspect the DataFrame for problematic columns
        print("DataFrame Info:")
        print(df.info())
        print("Sample Data:")
        print(df.head())

        # Flatten any recursive or nested structures
        df = df.applymap(lambda x: str(x) if isinstance(x, (dict, list, set)) else x)

        # Convert to JSON
        json_data = df.to_json(orient=orient, date_format=date_format)
        return json_data if json_data else "{}"  # Ensure non-None return
    except Exception as e:
        st.error(f"Error converting DataFrame to JSON: {e}")
        return "{}"  # Return empty JSON string on error

if __name__ == "__main__":
    main()