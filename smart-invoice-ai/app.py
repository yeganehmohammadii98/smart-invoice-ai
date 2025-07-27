import streamlit as st
import pandas as pd
import os
from datetime import datetime
from database.models import init_database, get_db_session, Invoice
from PIL import Image
import sys
import traceback
from utils.ocr_processor import OCRProcessor, get_tesseract_path
from utils.file_handler import FileHandler, get_file_preview_info
from database.models import OCRResult
import time
import streamlit as st
import pandas as pd
import os
import sys
import traceback
import time
from datetime import datetime
from database.models import init_database, get_db_session, Invoice, OCRResult, FieldExtraction, UserFeedback
from PIL import Image
from utils.ocr_processor import OCRProcessor, get_tesseract_path
from utils.file_handler import FileHandler, get_file_preview_info
from utils.field_extractor import FieldExtractor, calculate_field_confidence_score
from utils.learning_system import LearningSystem
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import os
import sys
import traceback
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from database.models import init_database, get_db_session, Invoice, OCRResult, FieldExtraction, UserFeedback
from PIL import Image

# Import your custom modules
try:
    from utils.ocr_processor import OCRProcessor, get_tesseract_path
    from utils.file_handler import FileHandler, get_file_preview_info
    from utils.field_extractor import FieldExtractor, calculate_field_confidence_score
    from utils.learning_system import LearningSystem
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Import plotting libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    st.warning("Plotly not installed. Installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    import plotly.express as px
    import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





# Page configuration
st.set_page_config(
    page_title="Smart Invoice AI System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize database
@st.cache_resource
def initialize_database():
    """Initialize database connection (cached to avoid repeated calls)"""
    return init_database()


def main():
    # Initialize database
    engine, SessionLocal = initialize_database()

    # App header
    st.title("🤖 Smart Invoice AI System")
    st.markdown("### Upload invoices and extract key information automatically")

    # Sidebar for navigation
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["Upload & Process", "View History", "Feedback & Corrections", "AI Learning Dashboard", "Model Statistics",
             "Settings"]
        )

    # Main content based on selected page
    if page == "Upload & Process":
        upload_page()
    elif page == "View History":
        history_page()
    elif page == "Feedback & Corrections":
        show_feedback_history()
    elif page == "AI Learning Dashboard":
        show_learning_dashboard()
    elif page == "Model Statistics":
        stats_page()
    elif page == "Settings":
        settings_page()
def upload_page():
    """Main upload and processing page"""
    st.header("📤 Upload Invoices")

    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("File Upload")

        # File uploader with multiple file support
        uploaded_files = st.file_uploader(
            "Choose invoice files",
            type=['pdf', 'png', 'jpg', 'jpeg', 'csv'],
            accept_multiple_files=True,
            help="Supported formats: PDF, PNG, JPG, CSV"
        )

        # Display upload instructions
        if not uploaded_files:
            st.info("👆 Upload your invoice files using the file picker above")
            st.markdown("""
            **Supported formats:**
            - 📄 PDF files (scanned or digital)
            - 🖼️ Image files (PNG, JPG, JPEG)
            - 📊 CSV files (structured data)

            **Tips:**
            - You can upload multiple files at once
            - Make sure images are clear and readable
            - For best results, use high-resolution scans
            """)

    # Replace the upload statistics section in upload_page() with this:
    with col2:
        st.subheader("Upload Statistics")

        db_session = get_db_session()
        try:
            total_invoices = db_session.query(Invoice).count()
            processed_today = db_session.query(Invoice).filter(
                Invoice.upload_date >= datetime.now().date()
            ).count()

            # Learning statistics with error handling
            try:
                learning_system = LearningSystem()
                learning_stats = learning_system.get_field_statistics()
                accuracy = learning_stats.get('accuracy_rate', 0.0) * 100
            except Exception as e:
                accuracy = 0.0
                logger.warning(f"Could not load learning statistics: {e}")

            st.metric("Total Invoices", total_invoices)
            st.metric("Processed Today", processed_today)
            st.metric("AI Accuracy", f"{accuracy:.1f}%")

        except Exception as e:
            st.error(f"Database error: {e}")
        finally:
            db_session.close()
    # Process uploaded files
    if uploaded_files:
        st.subheader("📋 Uploaded Files")

        # Display uploaded files in a nice format
        for i, uploaded_file in enumerate(uploaded_files):
            with st.expander(f"📎 {uploaded_file.name} ({uploaded_file.size} bytes)"):
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.write(f"**File type:** {uploaded_file.type}")
                    st.write(f"**Size:** {uploaded_file.size:,} bytes")

                with col2:
                    if st.button(f"Preview", key=f"preview_{i}"):
                        preview_file(uploaded_file)

                with col3:
                    if st.button(f"Process", key=f"process_{i}"):
                        process_file(uploaded_file)

        # Bulk actions
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("🔄 Process All", type="primary"):
                process_all_files(uploaded_files)

        with col2:
            if st.button("💾 Save All"):
                save_all_files(uploaded_files)


def preview_file(uploaded_file):
    """Enhanced preview with file information"""
    try:
        # Get file info
        file_info = get_file_preview_info(uploaded_file)

        # Display file information
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write(f"**📄 Name:** {file_info['name']}")
            st.write(f"**📊 Type:** {file_info['type']}")
        with col2:
            st.write(f"**💾 Size:** {file_info['size_mb']:.2f} MB")
            st.write(f"**📎 Extension:** {file_info['extension']}")

        # Show content preview
        if uploaded_file.type.startswith('image/'):
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Preview of {uploaded_file.name}", width=500)

            # Show image details
            st.info(f"📐 Dimensions: {image.size[0]} × {image.size[1]} pixels")

        elif uploaded_file.type == 'application/pdf':
            st.info("📄 PDF files will be converted to images for OCR processing")
            st.write("Click 'Process' to extract text using OCR")

        elif uploaded_file.type == 'text/csv':
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(10), use_container_width=True)
            st.info(f"📊 CSV contains {len(df)} rows and {len(df.columns)} columns")

    except Exception as e:
        st.error(f"Error previewing file: {e}")
        st.error(f"Error previewing file: {e}")


def process_file(uploaded_file):
    """Process a single file with OCR and field extraction"""

    # Validate file first
    file_handler = FileHandler()
    validation = file_handler.validate_file(uploaded_file)

    if not validation['valid']:
        st.error(f"❌ {validation['error']}")
        return

    with st.spinner(f"Processing {uploaded_file.name}..."):
        start_time = time.time()

        try:
            # Step 1: OCR Processing
            tesseract_path = get_tesseract_path()
            ocr_processor = OCRProcessor(tesseract_path)
            file_content = file_handler.get_file_content(uploaded_file)
            file_type = validation['file_type']

            ocr_result = ocr_processor.process_file(
                file_content, file_type, uploaded_file.name
            )

            if not ocr_result['success']:
                st.error(f"❌ OCR failed: {ocr_result.get('error', 'Unknown error')}")
                return

            # Step 2: Field Extraction with Learning
            field_extractor = FieldExtractor()

            # Apply previous learning patterns before extraction
            learning_system = LearningSystem()
            improved_extractor = learning_system.apply_learned_patterns(field_extractor, uploaded_file.name)

            extracted_fields = improved_extractor.extract_all_fields(ocr_result['text'])

            processing_time = time.time() - start_time

            # Step 3: Save to Database
            db_session = get_db_session()
            try:
                # Create invoice record
                new_invoice = Invoice(
                    filename=uploaded_file.name,
                    file_type=file_type,
                    processing_status='processed',
                    raw_text=ocr_result['text'][:10000],
                    invoice_number=extracted_fields.get('invoice_number', {}).get('value', ''),
                    invoice_date=extracted_fields.get('date', {}).get('value', ''),
                    supplier_name=extracted_fields.get('supplier', {}).get('value', ''),
                    total_amount=extracted_fields.get('total', {}).get('value', 0.0),
                    vat_amount=extracted_fields.get('vat', {}).get('value', 0.0),
                    confidence_invoice_number=extracted_fields.get('invoice_number', {}).get('confidence', 0.0),
                    confidence_date=extracted_fields.get('date', {}).get('confidence', 0.0),
                    confidence_supplier=extracted_fields.get('supplier', {}).get('confidence', 0.0),
                    confidence_total=extracted_fields.get('total', {}).get('confidence', 0.0)
                )

                db_session.add(new_invoice)
                db_session.flush()

                # Create OCR result record
                ocr_record = OCRResult(
                    invoice_id=new_invoice.id,
                    extracted_text=ocr_result['text'],
                    confidence_score=ocr_result['confidence'],
                    processing_time=processing_time,
                    ocr_method='tesseract',
                    pages_processed=ocr_result.get('pages', 1)
                )

                db_session.add(ocr_record)
                db_session.commit()

                # Step 4: Store in session state for full-page display
                st.session_state.current_processing_result = {
                    'invoice_id': new_invoice.id,
                    'filename': uploaded_file.name,
                    'extracted_fields': extracted_fields,
                    'full_text': ocr_result['text'],
                    'processing_time': processing_time,
                    'confidence': ocr_result['confidence'],
                    'pages': ocr_result.get('pages', 1)
                }

                # Success message
                st.success(f"✅ {uploaded_file.name} processed successfully!")

                # Show processing stats briefly
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Processing Time", f"{processing_time:.1f}s")
                with col2:
                    st.metric("OCR Confidence", f"{ocr_result['confidence']:.0%}")
                with col3:
                    st.metric("Pages", ocr_result.get('pages', 1))

                # Redirect message
                st.info("📋 **Scroll down to see the full extraction results and provide feedback!**")

            except Exception as e:
                st.error(f"Database error: {e}")
                db_session.rollback()
            finally:
                db_session.close()

        except Exception as e:
            st.error(f"Processing error: {e}")
            logger.error(f"Error processing {uploaded_file.name}: {traceback.format_exc()}")


def upload_page():
    """Enhanced upload page with full-page results display"""
    st.header("📤 Upload Invoices")

    # Upload section (top of page)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("File Upload")

        uploaded_files = st.file_uploader(
            "Choose invoice files",
            type=['pdf', 'png', 'jpg', 'jpeg', 'csv'],
            accept_multiple_files=True,
            help="Supported formats: PDF, PNG, JPG, CSV"
        )

        if not uploaded_files:
            st.info("👆 Upload your invoice files using the file picker above")
            st.markdown("""
            **Supported formats:**
            - 📄 PDF files (scanned or digital)
            - 🖼️ Image files (PNG, JPG, JPEG)
            - 📊 CSV files (structured data)

            **Tips:**
            - You can upload multiple files at once
            - Make sure images are clear and readable
            - For best results, use high-resolution scans
            """)

    with col2:
        st.subheader("Upload Statistics")

        db_session = get_db_session()
        try:
            total_invoices = db_session.query(Invoice).count()
            processed_today = db_session.query(Invoice).filter(
                Invoice.upload_date >= datetime.now().date()
            ).count()

            # Learning statistics
            learning_system = LearningSystem()
            learning_stats = learning_system.get_field_statistics()
            accuracy = learning_stats.get('accuracy_rate', 0.0) * 100

            st.metric("Total Invoices", total_invoices)
            st.metric("Processed Today", processed_today)
            st.metric("AI Accuracy", f"{accuracy:.1f}%")

        except Exception as e:
            st.error(f"Database error: {e}")
        finally:
            db_session.close()

    # Process uploaded files
    if uploaded_files:
        st.subheader("📋 Uploaded Files")

        for i, uploaded_file in enumerate(uploaded_files):
            with st.expander(f"📎 {uploaded_file.name} ({uploaded_file.size} bytes)"):
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.write(f"**File type:** {uploaded_file.type}")
                    st.write(f"**Size:** {uploaded_file.size:,} bytes")

                with col2:
                    if st.button(f"Preview", key=f"preview_{i}"):
                        preview_file(uploaded_file)

                with col3:
                    if st.button(f"Process", key=f"process_{i}"):
                        process_file(uploaded_file)

        # Bulk actions
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("🔄 Process All", type="primary"):
                process_all_files(uploaded_files)

        with col2:
            if st.button("💾 Save All"):
                save_all_files(uploaded_files)

    # FULL-PAGE RESULTS SECTION (separate from upload area)
    st.markdown("---")

    # Check if there's a processing result to display
    if 'current_processing_result' in st.session_state:
        result = st.session_state.current_processing_result

        st.markdown("## 🎯 Processing Results")
        display_field_extraction_interface(
            result['invoice_id'],
            result['filename'],
            result['extracted_fields'],
            result['full_text']
        )
    else:
        # Show placeholder when no results
        st.markdown("## 📋 Results Area")
        st.info("Upload and process an invoice above to see extraction results here.")

        # Show recent processing history
        show_recent_extractions()


def show_recent_extractions():
    """Show recent extractions for reference"""
    st.subheader("📚 Recent Extractions")

    db_session = get_db_session()
    try:
        recent_invoices = db_session.query(Invoice).order_by(
            Invoice.upload_date.desc()
        ).limit(5).all()

        if recent_invoices:
            for invoice in recent_invoices:
                with st.expander(f"📄 {invoice.filename} - {invoice.upload_date.strftime('%Y-%m-%d %H:%M')}"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write(f"**Invoice #:** {invoice.invoice_number or 'Not detected'}")
                        st.write(f"**Supplier:** {invoice.supplier_name or 'Not detected'}")
                    with col2:
                        st.write(f"**Date:** {invoice.invoice_date or 'Not detected'}")
                        st.write(f"**Total:** ${invoice.total_amount or 0:.2f}")
                    with col3:
                        st.write(f"**Status:** {invoice.processing_status}")
                        if st.button(f"View Details", key=f"view_{invoice.id}"):
                            # Load this invoice's results
                            load_invoice_results(invoice.id)
        else:
            st.info("No recent extractions found.")

    except Exception as e:
        st.error(f"Error loading recent extractions: {e}")
    finally:
        db_session.close()


def load_invoice_results(invoice_id):
    """Load and display results for a specific invoice"""
    db_session = get_db_session()
    try:
        invoice = db_session.query(Invoice).filter(Invoice.id == invoice_id).first()
        ocr_result = db_session.query(OCRResult).filter(OCRResult.invoice_id == invoice_id).first()

        if invoice and ocr_result:
            # Reconstruct extracted fields from database
            extracted_fields = {
                'invoice_number': {'value': invoice.invoice_number or '',
                                   'confidence': invoice.confidence_invoice_number or 0.0},
                'date': {'value': invoice.invoice_date or '', 'confidence': invoice.confidence_date or 0.0},
                'supplier': {'value': invoice.supplier_name or '', 'confidence': invoice.confidence_supplier or 0.0},
                'customer': {'value': 'Not stored', 'confidence': 0.0},  # Add to Invoice model if needed
                'total': {'value': invoice.total_amount or 0.0, 'confidence': invoice.confidence_total or 0.0},
                'subtotal': {'value': invoice.total_amount or 0.0, 'confidence': 0.8},  # Estimate
                'vat': {'value': invoice.vat_amount or 0.0, 'confidence': 0.8},
                'line_items_count': {'value': 0, 'confidence': 0.0},
                'line_items_subtotal': {'value': 0.0, 'confidence': 0.0}
            }

            # Update session state to show results
            st.session_state.current_processing_result = {
                'invoice_id': invoice.id,
                'filename': invoice.filename,
                'extracted_fields': extracted_fields,
                'full_text': ocr_result.extracted_text,
                'processing_time': ocr_result.processing_time,
                'confidence': ocr_result.confidence_score,
                'pages': ocr_result.pages_processed
            }

            st.experimental_rerun()  # Refresh page to show results

    except Exception as e:
        st.error(f"Error loading invoice results: {e}")
    finally:
        db_session.close()


def display_field_extraction_interface(invoice_id, filename, extracted_fields, full_text):
    """Display the field extraction interface with proper full-width layout"""

    st.markdown("---")
    st.header(f"📄 Processing: {filename}")
    st.markdown("**Method:** Rule-Based Pattern Matching")

    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📝 Extracted Fields", "📊 Confidence Analysis", "📄 Raw Text"])

    with tab1:
        st.subheader("Extracted Information")

        # Create form for editing extracted fields - FULL WIDTH
        with st.form(key=f"corrections_form_{invoice_id}"):

            # Create 2 columns for better layout
            col1, col2 = st.columns(2)

            corrected_fields = {}

            with col1:
                st.markdown("#### 📄 Document Details")

                # Invoice Number
                inv_num = extracted_fields.get('invoice_number', {})
                confidence = inv_num.get('confidence', 0.0)
                color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"

                st.markdown(f"{color} **Invoice Number** (Confidence: {confidence:.2f})")
                corrected_fields['invoice_number'] = st.text_input(
                    "Invoice Number:",
                    value=str(inv_num.get('value', '')),
                    key=f"inv_num_{invoice_id}",
                    help=f"Original extraction confidence: {confidence:.1%}"
                )

                # Date
                date_field = extracted_fields.get('date', {})
                confidence = date_field.get('confidence', 0.0)
                color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"

                st.markdown(f"{color} **Date** (Confidence: {confidence:.2f})")
                corrected_fields['date'] = st.text_input(
                    "Date:",
                    value=str(date_field.get('value', '')),
                    key=f"date_{invoice_id}",
                    help=f"Original extraction confidence: {confidence:.1%}"
                )

                # Supplier
                supplier_field = extracted_fields.get('supplier', {})
                confidence = supplier_field.get('confidence', 0.0)
                color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"

                st.markdown(f"{color} **Supplier** (Confidence: {confidence:.2f})")
                corrected_fields['supplier'] = st.text_input(
                    "Supplier:",
                    value=str(supplier_field.get('value', '')),
                    key=f"supplier_{invoice_id}",
                    help=f"Original extraction confidence: {confidence:.1%}"
                )

                # Customer
                customer_field = extracted_fields.get('customer', {})
                confidence = customer_field.get('confidence', 0.0)
                color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"

                st.markdown(f"{color} **Customer** (Confidence: {confidence:.2f})")
                corrected_fields['customer'] = st.text_input(
                    "Customer:",
                    value=str(customer_field.get('value', '')),
                    key=f"customer_{invoice_id}",
                    help=f"Original extraction confidence: {confidence:.1%}"
                )

            with col2:
                st.markdown("#### 💰 Financial Details")

                # Total
                total_field = extracted_fields.get('total', {})
                confidence = total_field.get('confidence', 0.0)
                color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"

                st.markdown(f"{color} **Total Amount** (Confidence: {confidence:.2f})")
                corrected_fields['total'] = st.number_input(
                    "Total Amount:",
                    value=float(total_field.get('value', 0.0)),
                    key=f"total_{invoice_id}",
                    help=f"Original extraction confidence: {confidence:.1%}",
                    format="%.2f"
                )

                # Subtotal
                subtotal_field = extracted_fields.get('subtotal', {})
                confidence = subtotal_field.get('confidence', 0.0)
                color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"

                st.markdown(f"{color} **Subtotal** (Confidence: {confidence:.2f})")
                corrected_fields['subtotal'] = st.number_input(
                    "Subtotal:",
                    value=float(subtotal_field.get('value', 0.0)),
                    key=f"subtotal_{invoice_id}",
                    help=f"Original extraction confidence: {confidence:.1%}",
                    format="%.2f"
                )

                # VAT
                vat_field = extracted_fields.get('vat', {})
                confidence = vat_field.get('confidence', 0.0)
                color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"

                st.markdown(f"{color} **VAT Amount** (Confidence: {confidence:.2f})")
                corrected_fields['vat'] = st.number_input(
                    "VAT Amount:",
                    value=float(vat_field.get('value', 0.0)),
                    key=f"vat_{invoice_id}",
                    help=f"Original extraction confidence: {confidence:.1%}",
                    format="%.2f"
                )

                # Currency (new field)
                st.markdown("🟡 **Currency** (Confidence: 0.50)")
                corrected_fields['currency'] = st.selectbox(
                    "Currency:",
                    options=["USD", "EUR", "GBP", "CAD", "AUD"],
                    index=0,
                    key=f"currency_{invoice_id}",
                    help="Currency detection will be improved in future versions"
                )

            # Line Items Section (Full Width)
            st.markdown("---")
            st.markdown("#### 📋 Line Items Details")

            col3, col4 = st.columns(2)

            with col3:
                # Line Items Count
                line_count_field = extracted_fields.get('line_items_count', {})
                confidence = line_count_field.get('confidence', 0.0)
                color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"

                st.markdown(f"{color} **Line Items Count** (Confidence: {confidence:.2f})")
                corrected_fields['line_items_count'] = st.number_input(
                    "Number of Line Items:",
                    value=int(line_count_field.get('value', 0)),
                    min_value=0,
                    max_value=100,
                    key=f"line_count_{invoice_id}",
                    help="Line item detection is being improved"
                )

            with col4:
                # Line Items Subtotal
                line_subtotal_field = extracted_fields.get('line_items_subtotal', {})
                confidence = line_subtotal_field.get('confidence', 0.0)
                color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"

                st.markdown(f"{color} **Line Items Subtotal** (Confidence: {confidence:.2f})")
                corrected_fields['line_items_subtotal'] = st.number_input(
                    "Line Items Subtotal:",
                    value=float(line_subtotal_field.get('value', 0.0)),
                    key=f"line_subtotal_{invoice_id}",
                    help="Line item subtotal detection is being improved",
                    format="%.2f"
                )

            # Submit section - Full Width
            st.markdown("---")

            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

            with col_btn1:
                submitted = st.form_submit_button(
                    "💾 Save Corrections & Learn",
                    type="primary",
                    use_container_width=True
                )

            with col_btn2:
                if st.form_submit_button(
                        "✅ Approve All",
                        use_container_width=True
                ):
                    st.success("✅ All extractions approved! Thank you for the feedback.")
                    save_approval_feedback(invoice_id, extracted_fields)

            with col_btn3:
                st.markdown("💡 **Tip:** Green = High confidence, Yellow = Medium, Red = Needs review")

            if submitted:
                handle_user_corrections(invoice_id, extracted_fields, corrected_fields)

    with tab2:
        st.subheader("📊 Confidence Analysis")

        # Show confidence metrics
        col1, col2 = st.columns([1, 1])

        with col1:
            create_confidence_chart(extracted_fields)

        with col2:
            # Confidence summary
            st.markdown("#### 📈 Confidence Summary")

            confidences = []
            for field_name, field_data in extracted_fields.items():
                if isinstance(field_data, dict) and 'confidence' in field_data:
                    confidences.append(field_data['confidence'])

            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                high_conf_count = len([c for c in confidences if c > 0.8])
                medium_conf_count = len([c for c in confidences if 0.5 <= c <= 0.8])
                low_conf_count = len([c for c in confidences if c < 0.5])

                st.metric("Average Confidence", f"{avg_confidence:.1%}")
                st.metric("High Confidence Fields", f"{high_conf_count}/{len(confidences)}")
                st.metric("Medium Confidence Fields", f"{medium_conf_count}/{len(confidences)}")
                st.metric("Low Confidence Fields", f"{low_conf_count}/{len(confidences)}")

                # Recommendations
                if avg_confidence > 0.8:
                    st.success("🎯 Excellent extraction quality!")
                elif avg_confidence > 0.6:
                    st.warning("⚠️ Good quality, some fields may need review")
                else:
                    st.error("❌ Low quality extraction, please review carefully")

    with tab3:
        st.subheader("📄 Raw Extracted Text")

        # Show full text with better formatting
        st.markdown("**Full OCR Text Output:**")

        # Add text statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Characters", len(full_text))
        with col2:
            word_count = len(full_text.split())
            st.metric("Words", word_count)
        with col3:
            line_count = len(full_text.split('\n'))
            st.metric("Lines", line_count)
        with col4:
            # Estimate reading time
            reading_time = max(1, word_count // 200)  # ~200 words per minute
            st.metric("Est. Reading Time", f"{reading_time} min")

        # Display text in expandable sections
        if len(full_text) > 3000:
            # Split long text into chunks
            chunk_size = 1500
            chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]

            for i, chunk in enumerate(chunks):
                with st.expander(f"📝 Text Section {i + 1} of {len(chunks)}"):
                    st.text_area(
                        f"Section {i + 1}:",
                        value=chunk,
                        height=200,
                        disabled=True,
                        key=f"text_chunk_{invoice_id}_{i}"
                    )
        else:
            # Show full text for shorter documents
            st.text_area(
                "Full extracted text:",
                value=full_text,
                height=400,
                disabled=True,
                key=f"full_text_{invoice_id}"
            )

        # Download option
        st.download_button(
            label="📥 Download Extracted Text",
            data=full_text,
            file_name=f"extracted_text_{filename}.txt",
            mime="text/plain"
        )


def save_approval_feedback(invoice_id, extracted_fields):
    """Save user approval feedback (when they click 'Approve All')"""
    try:
        db_session = get_db_session()

        # Save approval feedback for each field
        for field_name, field_data in extracted_fields.items():
            if isinstance(field_data, dict) and 'confidence' in field_data:
                feedback = UserFeedback(
                    invoice_id=invoice_id,
                    field_name=field_name,
                    original_value=str(field_data.get('value', '')),
                    corrected_value=str(field_data.get('value', '')),  # Same as original
                    feedback_type='confirmation',
                    confidence_before=field_data['confidence'],
                    user_rating=5,  # Approval = 5 stars
                    is_used_for_training=True
                )
                db_session.add(feedback)

        db_session.commit()

    except Exception as e:
        st.error(f"Error saving approval: {e}")
        db_session.rollback()
    finally:
        db_session.close()


def create_confidence_chart(extracted_fields):
    """Create confidence visualization chart with better styling"""

    if not PLOTLY_AVAILABLE:
        st.warning("📊 Chart visualization requires Plotly installation")
        return

    # Prepare data for chart
    field_names = []
    confidences = []
    colors = []

    field_display_names = {
        'invoice_number': 'Invoice Number',
        'date': 'Date',
        'supplier': 'Supplier',
        'customer': 'Customer',
        'total': 'Total',
        'subtotal': 'Subtotal',
        'vat': 'VAT',
        'line_items_count': 'Line Items Count',
        'line_items_subtotal': 'Line Items Subtotal'
    }

    for field_name, field_data in extracted_fields.items():
        if isinstance(field_data, dict) and 'confidence' in field_data:
            display_name = field_display_names.get(field_name, field_name.title())
            confidence = field_data['confidence']

            field_names.append(display_name)
            confidences.append(confidence)

            # Color based on confidence level
            if confidence > 0.8:
                colors.append('#28a745')  # Green
            elif confidence > 0.5:
                colors.append('#ffc107')  # Yellow
            else:
                colors.append('#dc3545')  # Red

    # Create bar chart using Plotly
    fig = go.Figure(data=[
        go.Bar(
            x=field_names,
            y=confidences,
            marker_color=colors,
            text=[f"{c:.0%}" for c in confidences],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.0%}<extra></extra>'
        )
    ])

    fig.update_layout(
        title="Field Extraction Confidence Scores",
        xaxis_title="Fields",
        yaxis_title="Confidence",
        yaxis=dict(range=[0, 1.0], tickformat='.0%'),
        height=400,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    fig.update_xaxes(tickangle=45)

    st.plotly_chart(fig, use_container_width=True)

def create_confidence_chart(extracted_fields: Dict):
    """Create confidence visualization chart"""

    # Prepare data for chart
    field_names = []
    confidences = []

    field_display_names = {
        'invoice_number': 'Customer',
        'date': 'Date',
        'supplier': 'Invoice Number',
        'customer': 'Line Item Count',
        'total': 'Subtotal',
        'subtotal': 'Supplier',
        'vat': 'Total',
        'line_items_count': 'Vat'
    }

    for field_name, field_data in extracted_fields.items():
        if isinstance(field_data, dict) and 'confidence' in field_data:
            display_name = field_display_names.get(field_name, field_name.title())
            field_names.append(display_name)
            confidences.append(field_data['confidence'])

    # Create bar chart using Plotly
    fig = go.Figure(data=[
        go.Bar(
            x=field_names,
            y=confidences,
            marker_color='#1f77b4',  # Blue color matching your design
            text=[f"{c:.2f}" for c in confidences],
            textposition='outside'
        )
    ])

    fig.update_layout(
        title="",
        xaxis_title="",
        yaxis_title="",
        yaxis=dict(range=[0, 1.0]),
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    fig.update_xaxes(tickangle=45)

    st.plotly_chart(fig, use_container_width=True)


def handle_user_corrections(invoice_id, original_fields, corrected_fields):
    """Handle user corrections and learn from them"""

    try:
        # Initialize learning system
        learning_system = LearningSystem()

        # Save corrections to database
        success = learning_system.save_field_corrections(
            invoice_id, original_fields, corrected_fields
        )

        if success:
            st.success("✅ Corrections saved successfully! The AI will learn from your feedback.")

            # Show learning statistics
            with st.expander("📊 Learning Impact"):
                try:
                    stats = learning_system.get_field_statistics()

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Extractions", stats.get('total_extractions', 0))
                    with col2:
                        st.metric("Total Corrections", stats.get('total_corrections', 0))
                    with col3:
                        accuracy = stats.get('accuracy_rate', 0.0) * 100
                        st.metric("Accuracy Rate", f"{accuracy:.1f}%")

                    # Show most problematic fields
                    problematic = stats.get('most_problematic_fields', [])
                    if problematic:
                        st.markdown("**Fields that need most improvement:**")
                        for field_name, error_count in problematic:
                            st.write(f"• {field_name.title()}: {error_count} corrections needed")

                except Exception as e:
                    st.warning(f"Could not load learning statistics: {e}")

            # Trigger relearning (placeholder for now)
            st.info("🧠 The AI model will be updated with your corrections in the next training cycle.")

        else:
            st.error("❌ Failed to save corrections. Please try again.")

    except Exception as e:
        st.error(f"Error processing corrections: {e}")
        logger.error(f"Error in handle_user_corrections: {e}")

def show_learning_dashboard():
    """Show comprehensive learning and improvement dashboard"""

    st.header("🧠 AI Learning Dashboard")

    try:
        learning_system = LearningSystem()

        # Get overall statistics
        stats = learning_system.get_field_statistics()
        patterns = learning_system.get_learning_patterns()

        # Overview metrics
        st.subheader("📊 Overall Performance")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Extractions", stats.get('total_extractions', 0))
        with col2:
            st.metric("Total Corrections", stats.get('total_corrections', 0))
        with col3:
            accuracy = stats.get('accuracy_rate', 0.0) * 100
            st.metric("Current Accuracy", f"{accuracy:.1f}%")
        with col4:
            trend = stats.get('improvement_trend', 'Unknown')
            st.metric("Trend", trend)

        # Field-specific accuracy
        st.subheader("🎯 Field-Specific Performance")

        if 'field_accuracy' in patterns and patterns['field_accuracy']:
            accuracy_data = patterns['field_accuracy']

            # Create accuracy chart
            fields = list(accuracy_data.keys())
            accuracies = [accuracy_data[field] * 100 for field in fields]

            fig = go.Figure(data=[
                go.Bar(
                    x=fields,
                    y=accuracies,
                    marker_color=['green' if acc > 80 else 'orange' if acc > 60 else 'red' for acc in accuracies],
                    text=[f"{acc:.1f}%" for acc in accuracies],
                    textposition='outside'
                )
            ])

            fig.update_layout(
                title="Field Extraction Accuracy",
                xaxis_title="Fields",
                yaxis_title="Accuracy (%)",
                yaxis=dict(range=[0, 100]),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No field accuracy data available yet. Process more invoices to see analytics.")

        # Most problematic fields
        st.subheader("⚠️ Areas for Improvement")
        problematic = stats.get('most_problematic_fields', [])

        if problematic:
            for i, (field_name, error_count) in enumerate(problematic, 1):
                with st.expander(f"{i}. {field_name.title()} - {error_count} corrections needed"):
                    st.write(f"**Field:** {field_name}")
                    st.write(f"**Error Count:** {error_count}")
                    st.write(f"**Recommendation:** Review extraction patterns for {field_name}")

                    # Show recent corrections for this field
                    db_session = get_db_session()
                    try:
                        recent_corrections = db_session.query(UserFeedback).filter(
                            UserFeedback.field_name == field_name,
                            UserFeedback.feedback_type == 'correction'
                        ).order_by(UserFeedback.feedback_date.desc()).limit(3).all()

                        if recent_corrections:
                            st.write("**Recent corrections:**")
                            for correction in recent_corrections:
                                st.write(f"• '{correction.original_value}' → '{correction.corrected_value}'")
                    except Exception as e:
                        st.write(f"Error loading corrections: {e}")
                    finally:
                        db_session.close()
        else:
            st.success("🎉 No major issues detected! The AI is performing well.")

        # Learning recommendations
        st.subheader("💡 Learning Recommendations")

        if accuracy < 70:
            st.warning("⚠️ **Low Overall Accuracy** - Consider retraining the model with more correction data")
        elif accuracy < 85:
            st.info("📈 **Good Progress** - Continue providing corrections to improve accuracy")
        else:
            st.success("🎯 **Excellent Performance** - The AI is learning well from your feedback")

        # Future improvements
        st.markdown("""
        **Planned Improvements:**
        - 🤖 Automatic model retraining based on corrections
        - 📝 Advanced pattern learning from user feedback  
        - 🧮 Machine learning model integration
        - 📊 Detailed confidence score improvements
        """)

    except Exception as e:
        st.error(f"Error loading learning dashboard: {e}")
        logger.error(f"Error in show_learning_dashboard: {e}")

def show_extraction_results_and_feedback(invoice_data):
    pass


def display_processing_results(invoice_data, ocr_result):
    """Display processing results without nesting issues"""

    if ocr_result['success']:
        st.success(f"✅ {invoice_data['filename']} processed successfully!")

        # Show processing stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Processing Time", f"{invoice_data['processing_time']:.1f}s")
        with col2:
            st.metric("Confidence", f"{invoice_data['confidence']:.0%}")
        with col3:
            st.metric("Pages", invoice_data['pages'])

        # Show extracted text and feedback form
        st.markdown("---")
        show_extraction_results_and_feedback(invoice_data)

    else:
        st.error(f"❌ Error processing {invoice_data['filename']}: {ocr_result.get('error', 'Unknown error')}")

def process_all_files(uploaded_files):
    """Process all uploaded files"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        process_file(uploaded_file)
        progress_bar.progress((i + 1) / len(uploaded_files))

    status_text.text("✅ All files processed!")
    st.balloons()


def save_all_files(uploaded_files):
    """Save all files to uploads directory"""
    os.makedirs("uploads", exist_ok=True)
    saved_count = 0

    for uploaded_file in uploaded_files:
        try:
            # Save file to uploads directory
            file_path = os.path.join("uploads", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_count += 1
        except Exception as e:
            st.error(f"Error saving {uploaded_file.name}: {e}")

    st.success(f"💾 Saved {saved_count} files to uploads directory")


def history_page():
    """Display processing history"""
    st.header("📊 Processing History")

    db_session = get_db_session()
    try:
        # Get all invoices from database
        invoices = db_session.query(Invoice).order_by(Invoice.upload_date.desc()).all()

        if invoices:
            # Convert to DataFrame for display
            data = []
            for invoice in invoices:
                data.append({
                    'ID': invoice.id,
                    'Filename': invoice.filename,
                    'Upload Date': invoice.upload_date.strftime('%Y-%m-%d %H:%M'),
                    'File Type': invoice.file_type,
                    'Status': invoice.processing_status,
                    'Invoice Number': invoice.invoice_number or 'Not extracted',
                    'Supplier': invoice.supplier_name or 'Not extracted',
                    'Total Amount': f"${invoice.total_amount:.2f}" if invoice.total_amount else 'Not extracted'
                })

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            # Summary statistics
            st.subheader("📈 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Files", len(invoices))
            with col2:
                processed = len([i for i in invoices if i.processing_status == 'processed'])
                st.metric("Processed", processed)
            with col3:
                pending = len([i for i in invoices if i.processing_status == 'pending'])
                st.metric("Pending", pending)
            with col4:
                errors = len([i for i in invoices if i.processing_status == 'error'])
                st.metric("Errors", errors)
        else:
            st.info("No invoices processed yet. Go to 'Upload & Process' to get started!")

    except Exception as e:
        st.error(f"Database error: {e}")
    finally:
        db_session.close()


def stats_page():
    """Display model statistics"""
    st.header("🤖 Model Statistics")
    st.info("Model performance metrics will be available after implementing auto fine-tuning in Phase 6")

    # Placeholder content
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Model Performance")
        st.metric("Overall Accuracy", "Coming soon")
        st.metric("Invoice Number Accuracy", "Coming soon")
        st.metric("Date Extraction Accuracy", "Coming soon")
        st.metric("Amount Extraction Accuracy", "Coming soon")

    with col2:
        st.subheader("Training Progress")
        st.metric("Training Samples", "Coming soon")
        st.metric("Model Version", "Coming soon")
        st.metric("Last Retrain Date", "Coming soon")


def settings_page():
    """Application settings"""
    st.header("⚙️ Settings")

    st.subheader("Database Settings")
    if st.button("🔄 Reset Database"):
        if st.checkbox("I understand this will delete all data"):
            # Reset database (placeholder)
            st.warning("Database reset functionality will be implemented in later phases")

    st.subheader("Model Settings")
    confidence_threshold = st.slider(
        "Minimum Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="Fields with confidence below this threshold will be flagged for review"
    )

    st.subheader("Upload Settings")
    max_file_size = st.number_input(
        "Maximum File Size (MB)",
        min_value=1,
        max_value=100,
        value=10
    )

    if st.button("💾 Save Settings"):
        st.success("Settings saved! (Note: Settings persistence will be implemented in later phases)")


def show_extraction_results_and_feedback(invoice_data):
    """Show extracted text and allow user feedback"""

    st.subheader("📄 Extraction Results & Feedback")

    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📝 Extracted Text", "✏️ Provide Feedback", "📊 Field Extraction"])

    with tab1:
        st.markdown("**Full Extracted Text:**")

        # Show text in a scrollable area
        if len(invoice_data['extracted_text']) > 2000:
            st.text_area(
                "Extracted Text:",
                invoice_data['extracted_text'],
                height=300,
                disabled=True,
                key=f"text_display_{invoice_data['id']}"
            )
            st.info(f"📝 Total characters: {len(invoice_data['extracted_text'])}")
        else:
            st.text_area(
                "Extracted Text:",
                invoice_data['extracted_text'],
                height=200,
                disabled=True,
                key=f"text_display_short_{invoice_data['id']}"
            )

    with tab2:
        show_feedback_form(invoice_data)

    with tab3:
        show_field_extraction_preview(invoice_data)


def show_feedback_form(invoice_data):
    """Show feedback form for user corrections"""

    st.markdown("**Help improve the AI by providing feedback:**")

    # Overall quality rating
    col1, col2 = st.columns([1, 2])
    with col1:
        quality_rating = st.select_slider(
            "Overall extraction quality:",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: "⭐" * x,
            key=f"quality_{invoice_data['id']}"
        )

    with col2:
        feedback_type = st.selectbox(
            "Feedback type:",
            ["Excellent - No corrections needed",
             "Good - Minor corrections",
             "Fair - Some corrections needed",
             "Poor - Major corrections needed"],
            key=f"feedback_type_{invoice_data['id']}"
        )

    # Text correction area
    st.markdown("**Correct the extracted text if needed:**")
    corrected_text = st.text_area(
        "Edit the text below to correct any errors:",
        value=invoice_data['extracted_text'],
        height=200,
        key=f"correction_{invoice_data['id']}"
    )

    # Additional feedback
    user_notes = st.text_area(
        "Additional notes (optional):",
        placeholder="Any specific issues or suggestions...",
        key=f"notes_{invoice_data['id']}"
    )

    # Submit feedback
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("💾 Save Feedback", key=f"save_feedback_{invoice_data['id']}"):
            save_user_feedback(invoice_data, quality_rating, feedback_type, corrected_text, user_notes)

    with col2:
        if st.button("👍 Looks Good", key=f"approve_{invoice_data['id']}"):
            save_user_feedback(invoice_data, 5, "Excellent - No corrections needed", invoice_data['extracted_text'],
                               "User approved extraction")


def show_field_extraction_preview(invoice_data):
    """Show preview of field extraction (placeholder for Phase 3)"""

    st.markdown("**🔮 Intelligent Field Extraction Preview**")
    st.info("🚀 Smart field extraction will be implemented in Phase 3!")

    # Show what fields we'll extract
    fields_to_extract = [
        "📄 Invoice Number",
        "📅 Invoice Date",
        "🏢 Supplier Name",
        "💰 Total Amount",
        "💱 Currency",
        "📊 VAT Amount"
    ]

    st.markdown("**Fields that will be automatically extracted:**")
    for field in fields_to_extract:
        st.markdown(f"- {field}")

    # Simple regex preview for demo
    text = invoice_data['extracted_text'].upper()

    # Very basic field detection (improved in Phase 3)
    st.markdown("**🔍 Basic Pattern Detection:**")

    import re

    # Look for invoice numbers
    invoice_patterns = re.findall(r'(?:INVOICE|INV|#)\s*:?\s*([A-Z0-9-]+)', text)
    if invoice_patterns:
        st.success(f"📄 Possible Invoice Number: {invoice_patterns[0]}")

    # Look for amounts
    amount_patterns = re.findall(r'(?:TOTAL|AMOUNT|SUM)\s*:?\s*\$?([0-9,]+\.?[0-9]*)', text)
    if amount_patterns:
        st.success(f"💰 Possible Total Amount: ${amount_patterns[0]}")

    # Look for dates
    date_patterns = re.findall(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text)
    if date_patterns:
        st.success(f"📅 Possible Date: {date_patterns[0]}")


def save_user_feedback(invoice_data, rating, feedback_type, corrected_text, notes):
    """Save user feedback to database"""

    try:
        db_session = get_db_session()

        # Check if text was corrected
        text_corrected = corrected_text.strip() != invoice_data['extracted_text'].strip()
        correction_count = 1 if text_corrected else 0

        # Save to UserFeedback table
        feedback = UserFeedback(
            invoice_id=invoice_data['id'],
            field_name='full_text',
            original_value=invoice_data['extracted_text'],
            corrected_value=corrected_text,
            feedback_type='correction' if text_corrected else 'confirmation',
            confidence_before=invoice_data['confidence'],
            user_rating=rating,
            is_used_for_training=False  # Will be used in auto-training
        )

        db_session.add(feedback)

        # Update invoice record
        invoice = db_session.query(Invoice).filter(Invoice.id == invoice_data['id']).first()
        if invoice and text_corrected:
            invoice.raw_text = corrected_text[:10000]  # Update with corrected text

        db_session.commit()

        if text_corrected:
            st.success("✅ Thank you! Your corrections have been saved and will help improve the AI.")
        else:
            st.success("✅ Thank you for confirming the extraction quality!")

        # Update session state
        for item in st.session_state.processed_invoices:
            if item['id'] == invoice_data['id']:
                item['user_feedback_provided'] = True
                break

    except Exception as e:
        st.error(f"Error saving feedback: {e}")
        db_session.rollback()
    finally:
        db_session.close()


def show_feedback_history():
    """Show historical corrections and feedback"""

    st.header("📊 Feedback & Corrections History")

    db_session = get_db_session()
    try:
        # Get all feedback
        feedbacks = db_session.query(UserFeedback).order_by(UserFeedback.feedback_date.desc()).all()

        if feedbacks:
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)

            total_feedback = len(feedbacks)
            corrections = len([f for f in feedbacks if f.feedback_type == 'correction'])
            avg_rating = sum([f.user_rating for f in feedbacks if f.user_rating]) / len(
                [f for f in feedbacks if f.user_rating])

            with col1:
                st.metric("Total Feedback", total_feedback)
            with col2:
                st.metric("Corrections Made", corrections)
            with col3:
                st.metric("Average Rating", f"{avg_rating:.1f}⭐")
            with col4:
                improvement_rate = (corrections / total_feedback * 100) if total_feedback > 0 else 0
                st.metric("Improvement Rate", f"{improvement_rate:.1f}%")

            # Show feedback details
            st.subheader("📝 Recent Feedback")

            for feedback in feedbacks[:10]:  # Show last 10
                with st.expander(f"Feedback #{feedback.id} - {feedback.feedback_date.strftime('%Y-%m-%d %H:%M')}"):
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.write(f"**Type:** {feedback.feedback_type}")
                        st.write(f"**Rating:** {'⭐' * (feedback.user_rating or 0)}")
                        st.write(f"**Field:** {feedback.field_name}")
                        st.write(f"**Confidence Before:** {feedback.confidence_before:.0%}")

                    with col2:
                        if feedback.feedback_type == 'correction':
                            st.write("**Original Text:**")
                            st.code(feedback.original_value[:200] + "..." if len(
                                feedback.original_value) > 200 else feedback.original_value)
                            st.write("**Corrected Text:**")
                            st.code(feedback.corrected_value[:200] + "..." if len(
                                feedback.corrected_value) > 200 else feedback.corrected_value)
        else:
            st.info("No feedback provided yet. Process some invoices and provide feedback to see history here!")

    except Exception as e:
        st.error(f"Error loading feedback history: {e}")
    finally:
        db_session.close()
if __name__ == "__main__":
    main()

