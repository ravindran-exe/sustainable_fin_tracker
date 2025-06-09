
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import random
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import streamlit as st
import io

class SyntheticBankStatementGenerator:
    def __init__(self):
        self.fake = Faker()
        self.account_details = {}
        self.transaction_df = None
        self.include_category = True
        self.show_decimals = True
        self.include_large_txns = True

    def generate_account_details(self, currency="USD"):
        self.account_details = {
            'account_name': self.fake.name(),
            'account_number': self.fake.bban(),
            'bank_name': f"{self.fake.company()} Bank",
            'branch': f"{self.fake.city()} Branch",
            'ifsc_code': self.fake.bothify(text='????0####??'),
            'currency': currency,
            'opening_balance': round(random.uniform(1000, 10000), 2)
        }
        return self.account_details

    def generate_transactions(self, start_date, end_date, transaction_count=50, custom_categories=None, include_category=True, include_large_txns=True):
        self.include_category = include_category
        self.include_large_txns = include_large_txns

        if not self.account_details:
            self.generate_account_details()

        dates = [start_date + timedelta(days=random.randint(0, (end_date - start_date).days)) 
                 for _ in range(transaction_count)]
        dates.sort()

        categories = custom_categories if custom_categories else [
            'Groceries', 'Restaurant', 'Transport', 'Entertainment', 
            'Shopping', 'Utilities', 'Salary', 'Transfer', 'Investment', 'Cash'
        ]

        transactions = []
        balance = self.account_details['opening_balance']

        for date in dates:
            is_credit = random.random() < 0.3
            category = random.choice(categories)

            max_credit = 5000 if include_large_txns else 1500
            max_debit = 500 if include_large_txns else 300

            if is_credit:
                amount = round(random.uniform(100, max_credit), 2)
                particulars = self._shorten(self._generate_credit_particulars(category))
                debit = 0
                credit = amount
            else:
                amount = round(random.uniform(10, max_debit), 2)
                particulars = self._shorten(self._generate_debit_particulars(category))
                debit = amount
                credit = 0

            balance += credit - debit

            transaction = {
                'Date': date,
                'Particulars': particulars,
                'Debit': debit,
                'Credit': credit,
                'Balance': balance
            }

            if include_category:
                transaction['Category'] = category

            transactions.append(transaction)

        self.transaction_df = pd.DataFrame(transactions)
        return self.transaction_df

    def _shorten(self, text, max_length=40):
        return (text[:max_length] + '...') if len(text) > max_length else text

    def _generate_debit_particulars(self, category):
        if category == 'Groceries':
            return f"POS/DMART/BILL#{self.fake.bothify(text='D####')}"
        elif category == 'Restaurant':
            return f"UPI/ZOMATO/ORDER#{self.fake.bothify(text='Z####')}"
        elif category == 'Transport':
            return f"UPI/OLA/RIDE/{self.fake.random_number(digits=5)}"
        elif category == 'Entertainment':
            return f"NETFLIX/SUBSCRIPTION"
        elif category == 'Shopping':
            return f"AMAZON/ORDER#{self.fake.bothify(text='A#####')}"
        elif category == 'Utilities':
            return f"RECHARGE/PHONEPE"
        elif category == 'Transfer':
            return f"IMPS/TO/{self.fake.first_name().upper()}"
        elif category == 'Cash':
            return f"CASH WITHDRAWAL"
        else:
            return f"{category.upper()} TRANSACTION"

    def _generate_credit_particulars(self, category):
        if category == 'Salary':
            return f"SALARY/{self.fake.company().upper()}"
        elif category == 'Transfer':
            return f"NEFT/{self.fake.company().upper()}"
        elif category == 'Investment':
            return f"CRED RETURN"
        elif category == 'Cash':
            return f"CASH DEPOSIT"
        else:
            return f"{category.upper()} CREDIT"

    def generate_pdf_statement(self, show_decimals=True):
        self.show_decimals = show_decimals
        if self.transaction_df is None:
            raise ValueError("Generate transactions first!")

        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        elements = []

        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading_style = styles['Heading2']
        normal_style = styles['Normal']

        elements.append(Paragraph(f"{self.account_details['bank_name']}", title_style))
        elements.append(Paragraph("Bank Statement", heading_style))
        elements.append(Spacer(1, 0.25 * inch))

        account_info = [
            ["Account Name:", self.account_details['account_name']],
            ["Account Number:", self.account_details['account_number']],
            ["Branch:", self.account_details['branch']],
            ["Currency:", self.account_details['currency']],
            ["Statement Period:", f"{self.transaction_df['Date'].min().strftime('%d-%b-%Y')} to {self.transaction_df['Date'].max().strftime('%d-%b-%Y')}"]
        ]

        account_table = Table(account_info, colWidths=[1.5 * inch, 4 * inch])
        account_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(account_table)
        elements.append(Spacer(1, 0.5 * inch))

        opening_balance = self.account_details['opening_balance']
        closing_balance = self.transaction_df['Balance'].iloc[-1]
        total_debits = self.transaction_df['Debit'].sum()
        total_credits = self.transaction_df['Credit'].sum()

        fmt = "{:,.2f}" if show_decimals else "{:,.0f}"

        summary_info = [
            ["Opening Balance:", f"{self.account_details['currency']} {fmt.format(opening_balance)}"],
            ["Total Debits:", f"{self.account_details['currency']} {fmt.format(total_debits)}"],
            ["Total Credits:", f"{self.account_details['currency']} {fmt.format(total_credits)}"],
            ["Closing Balance:", f"{self.account_details['currency']} {fmt.format(closing_balance)}"]
        ]

        summary_table = Table(summary_info, colWidths=[1.5 * inch, 1.5 * inch])
        summary_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 0.5 * inch))

        elements.append(Paragraph("Transaction Details", heading_style))
        elements.append(Spacer(1, 0.25 * inch))

        headers = ['Date', 'Description', 'Debit', 'Credit', 'Balance']
        colWidths = [0.8 * inch, 3 * inch, 1 * inch, 1 * inch, 1 * inch]

        if self.include_category:
            headers.append('Category')
            colWidths.append(1.2 * inch)

        transaction_data = [headers]
        for _, row in self.transaction_df.iterrows():
            row_data = [
                row['Date'].strftime('%d-%b-%Y'),
                row['Particulars'],
                f"{self.account_details['currency']} {fmt.format(row['Debit'])}" if row['Debit'] > 0 else "",
                f"{self.account_details['currency']} {fmt.format(row['Credit'])}" if row['Credit'] > 0 else "",
                f"{self.account_details['currency']} {fmt.format(row['Balance'])}"
            ]
            if self.include_category:
                row_data.append(row.get('Category', ''))
            transaction_data.append(row_data)

        transaction_table = Table(transaction_data, colWidths=colWidths, repeatRows=1)
        transaction_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#D9E1F2')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#8EA9DB')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (2, 1), (4, -1), 'RIGHT'),
        ]))
        elements.append(transaction_table)

        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%d-%b-%Y %H:%M:%S')}", normal_style))
        elements.append(Paragraph("This is a computer-generated statement. No signature required.", normal_style))

        doc.build(elements)
        pdf_buffer.seek(0)
        return pdf_buffer

def main():
    st.set_page_config(page_title="Synthetic Bank Statement Generator", page_icon="🏦", layout="wide")

    st.title("🏦 Synthetic Bank Statement Generator")
    st.markdown("Generate customizable fake bank statements for testing or demonstration purposes.")

    with st.sidebar:
        st.header("Settings")
        currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "INR", "CAD", "AUD"])
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        end_date = st.date_input("End Date", datetime.now())
        transaction_count = st.slider("Number of Transactions", 10, 500, 50)
        custom_categories = st.text_input("Custom Categories (comma-separated)", "Groceries,Restaurant,Transport,Entertainment,Shopping,Utilities,Salary,Transfer,Investment,Cash")
        include_category = st.checkbox("Include Category Column", value=True)
        show_decimals = st.checkbox("Include Decimal Values in Output", value=True)
        include_large_txns = st.checkbox("Include Large Value Transactions", value=True)

    if st.button("Generate Statement"):
        generator = SyntheticBankStatementGenerator()
        generator.generate_account_details(currency=currency)

        categories = [c.strip() for c in custom_categories.split(",")] if custom_categories else None

        with st.spinner("Generating transactions..."):
            transactions = generator.generate_transactions(
                start_date=start_date,
                end_date=end_date,
                transaction_count=transaction_count,
                custom_categories=categories,
                include_category=include_category,
                include_large_txns=include_large_txns
            )

        st.success("✅ Transactions generated successfully!")
        st.dataframe(transactions.head(10))

        with st.spinner("Creating PDF..."):
            pdf_buffer = generator.generate_pdf_statement(show_decimals=show_decimals)

        st.download_button(
            label="📥 Download PDF Statement",
            data=pdf_buffer,
            file_name="bank_statement.pdf",
            mime="application/pdf"
        )

        export_df = transactions.copy()
        if not show_decimals:
            for col in ['Debit', 'Credit', 'Balance']:
                export_df[col] = export_df[col].apply(lambda x: int(round(x)) if x != 0 else 0)

        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="transactions.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()