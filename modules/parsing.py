import re
import json
import pandas as pd
import streamlit as st
import google.generativeai as genai

# Define the model and prompt
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

def parse_transactions_advanced(text, tables_data=None):
    """Advanced transaction parsing with better JSON handling and validation"""
    try:
        # Strategy 1: Use AI model to parse transactions
        response = model.generate_content(prompt.format(text=text))
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
    """Clean and validate transaction data with better number parsing"""
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
        
        # Validate transaction logic - if both debit and credit are > 0, it's likely an error
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
    """Fallback regex-based transaction parsing with better patterns"""
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