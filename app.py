import streamlit as st
import google.generativeai as genai
import pandas as pd
import io
import easyocr
import cv2
import numpy as np
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Configure API Key
GENAI_API_KEY = "AIzaSyCUzEyQTnzmSUqLaSa1rYE1BFW5mmk8loA"  # Replace with your actual API key
genai.configure(api_key=GENAI_API_KEY)

# Default values
DEFAULTS = {
    "SE": {
        "lectures": [
            ("AOA", "PCA"),
            ("DBMS", "RSG"),
            ("OS", "SM"),
            ("MP", "DGJ"),
            ("EM", "PS")
        ],
        "labs": [
            ("AOAL", "PCA"),
            ("DBMSL", "RSG"),
            ("OSL", "SM"),
            ("MPLL", "DGJ"),
            ("EML", "PS")
         ]
    },
    "TE": {
        "lectures": [
            ("DAV", "SAP"),
            ("CSS", "DGJ"),
            ("SEPM", "NAS"),
            ("ML", "RAS"),
            ("IVP", "SV")
        ],
        "labs": [
            ("DAVL", "SAP"),
            ("CSSL", "DGJ"),
            ("SEPML", "NAS"),
            ("CCL", "SV"),
            ("IVPL", "SV")
        ]
    },
    "BE": {
        "lectures": [
            ("SMA", "PCA"),
            ("AAl", "MPJ"),
            ("RN", "DI")
        ],
        "labs": [
            ("SMAL", "PCA"),
            ("AAIL", "MPJ"),
            ("RNL", "DI")
        ]
    }
}

# Time Slots
TIME_SLOTS = [
    '9:15-10:15', '10:15-11:15', '11:15-12:15', 
    '12:15-1:05', '1:35-2:35', '2:35-3:35'
]

# Venues
CLASSROOMS = ['CR1', 'CR2']
LABS = ['LAB1', 'LAB2', 'LAB3']

def extract_text_from_image(uploaded_file):
    """Extract text from an uploaded image file using EasyOCR."""
    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])
    
    # Read the image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Convert from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform text detection and recognition
    results = reader.readtext(image)
    
    # Draw bounding boxes and text on the image
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        
        # Draw rectangle
        cv2.rectangle(image_rgb, top_left, bottom_right, (0, 255, 0), 2)
        
        # Put text
        cv2.putText(image_rgb, text, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Extract just the text parts
    extracted_text = [text for (_, text, _) in results]
    
    return extracted_text, image_rgb

def generate_timetable_with_ai(subjects_faculty):
    """Generate timetable using Gemini API with structured output"""
    prompt = f"""
    Generate a comprehensive weekly timetable for SE, TE, and BE divisions with:
    IMPORTANT: You MUST format the output as proper markdown tables with clear borders.
    Each division's timetable should be in a separate markdown table with these exact headers:
    
    ### SE/TE/BE DIVISION TIMETABLE ###
    | Time       | Monday         | Tuesday        | Wednesday      | Thursday       | Friday         |
    |------------|----------------|----------------|----------------|----------------|----------------|
    | 9:15-10:15 | Subject(Fac) Venue | ... | ... | ... | ... |
    | ...        | ...            | ...            | ...            | ...            | ...            |
    
    Rules:
    1. Use exactly the format above with pipe characters and dashes for table borders
    2. Include all time slots for each day
    3. For each cell, show Subject(Faculty) Venue or Lab(Faculty) Venue
    4. Ensure no time/resource conflicts exist
    
    Scheduling Constraints:
    - Lectures: 1 hour slots (9:15-10:15, 10:15-11:15, 11:15-12:15, 12:15-1:05, 1:35-2:35, 2:35-3:35), max 2 concurrent lectures (CR1 + CR2)
    - Labs: max 3 concurrent labs
    - BE division ends before lunch (no afternoon classes)
    - ML RAS faculty cannot have 9:15am classes
    
    Input Data (from DEFAULTS):
    {DEFAULTS}
    """
    
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text

def parse_ai_timetable(timetable_text):
    """Parse the AI-generated timetable into DataFrames"""
    divisions = {}
    current_division = None
    lines = [line.strip() for line in timetable_text.split('\n') if line.strip()]
    
    for line in lines:
        if "DIVISION TIMETABLE" in line:
            current_division = line.split()[0]
            divisions[current_division] = []
        elif current_division and line.startswith('|'):
            # Remove any markdown formatting that might interfere
            clean_line = line.replace('**', '').replace('__', '')
            parts = [p.strip() for p in clean_line.split('|')[1:-1]]
            if len(parts) > 1:  # At least Time + one day
                divisions[current_division].append(parts)
    
    # Convert to DataFrames
    timetable_dfs = {}
    for div, data in divisions.items():
        if data and len(data) > 1:
            # First row is headers (days)
            columns = data[0][1:]
            # Subsequent rows are time slots and data
            times = []
            rows = []
            for row in data[1:]:
                if len(row) > 1:
                    times.append(row[0])
                    rows.append(row[1:len(columns)+1])
            
            if times and rows:
                timetable_dfs[div] = pd.DataFrame(rows, columns=columns, index=times)
                timetable_dfs[div].index.name = 'Time'
    
    return timetable_dfs

def generate_pdf(timetables):
    """Generate PDF with timetables for all divisions"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = []
    
    for division, timetable in timetables.items():
        # Add division title
        story.append(Paragraph(f"<b>{division} Division Timetable</b>", styles['Title']))
        
        # Convert DataFrame to list of lists for PDF table
        table_data = [['Time'] + list(timetable.columns)]
        for time, row in timetable.iterrows():
            table_data.append([time] + list(row))
        
        # Create table
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        
        story.append(t)
        story.append(Paragraph("<br/><br/>", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def main():
    st.title("📅 AI-Powered College Timetable Generator")
    
    st.markdown("""
    This tool can either:
    1. Extract text from an image of a timetable/schedule using OCR
    2. Generate optimized timetables using AI
    """)
    
    # Image upload section
    st.header("Step 1: Upload Image for Text Extraction")
    uploaded_file = st.file_uploader("Upload an image of your timetable/schedule", type=["jpg", "jpeg", "png"])
    
    extracted_text = None
    extracted_image = None
    
    if uploaded_file is not None:
        with st.spinner("Extracting text from image..."):
            try:
                extracted_text, extracted_image = extract_text_from_image(uploaded_file)
                
                st.success("Text extracted successfully!")
                
                # Display extracted text
                st.subheader("Extracted Text:")
                st.text("\n".join(extracted_text))
                
                # Display image with bounding boxes
                st.subheader("Image with Detected Text:")
                st.image(extracted_image, caption="Text Detection Results", use_column_width=True)
                
            except Exception as e:
                st.error(f"Error during text extraction: {str(e)}")
    
    # Timetable generation section (only show if text was extracted)
    if extracted_text:
        st.header("Step 2: Generate Timetable from Extracted Text")
        
        
        
        # Generate Timetables button
        if st.button("🔄 Generate Timetables from Extracted Text"):
            with st.spinner("Generating optimized timetables using AI..."):
                try:
                    # Use the extracted text as input
                    formatted_input = "\n".join(extracted_text)
                    
                    # Generate with AI
                    timetable_text = generate_timetable_with_ai(formatted_input)
                    
                    # Display raw output first
                    with st.expander("View Raw AI Output", expanded=True):
                        st.markdown(timetable_text)
                    
                    # Parse into DataFrames
                    timetables = parse_ai_timetable(timetable_text)
                    
                    if timetables:
                        # Display Timetables
                        for division in DEFAULTS.keys():
                            if division in timetables:
                                st.subheader(f"{division} Division Timetable")
                                st.table(timetables[division])
                        
                        # PDF Generation
                        pdf_buffer = generate_pdf(timetables)
                        st.download_button(
                            label="📄 Download as PDF",
                            data=pdf_buffer,
                            file_name=f"timetable_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.error("Failed to parse timetables from AI output. Please check the raw output above.")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please try again or check the API key.")

if __name__ == "__main__":
    main()