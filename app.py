import streamlit as st
import openai
import cv2
import os
import re
import pytesseract
import pandas as pd
import numpy as np
from PIL import Image
import base64


# Initialize OpenAI API
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Initialize an empty list to store details
details_list = []

# Streamlit sidebar and main area
st.title('Business Card OCR App')

# Upload images through Streamlit
uploaded_files = st.file_uploader("Choose images...", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

# Process each uploaded file
for uploaded_file in uploaded_files:
    image = Image.open(uploaded_file)
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # OCR with Tesseract
    extracted_text = pytesseract.image_to_string(image_cv2)

    # Send OCR text to OpenAI API for parsing
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Extract the name (full name), company, email, phone number, and address from the following text: {extracted_text}",
        max_tokens=150
    )
    
    # Extract fields from API response
    parsed_text = response.choices[0].text.strip()
    name = re.findall(r"Name: (.+)", parsed_text)
    company = re.findall(r"Company: (.+)", parsed_text)
    email = re.findall(r"Email: (.+)", parsed_text)
    phone_number = re.findall(r"Phone Number: (.+)", parsed_text)
    address = re.findall(r"Address: (.+)", parsed_text)

    # Append details to the list
    details_list.append({
        'Name': name[0] if name else '',
        'Company': company[0] if company else '',
        'Email': email[0] if email else '',
        'Phone Number': phone_number[0] if phone_number else '',
        'Address': address[0] if address else ''
    })

# Convert list of dictionaries to DataFrame
details_df = pd.DataFrame(details_list)

# Function to make dataframe downloadable
def to_csv_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() 
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV</a>'

if details_list:
    st.write("Here are the extracted details:")
    st.write(details_df)
    
    # Generate download link for the CSV
    st.markdown(to_csv_download_link(details_df, "business_card_details"), unsafe_allow_html=True)
