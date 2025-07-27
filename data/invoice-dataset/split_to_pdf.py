import pandas as pd
import fitz  # PyMuPDF
import os


def split_csv_to_files(input_csv, output_dir='/home/yeganeh/PycharmProjects/Inr_task/data/invoice-dataset/output', rows_per_file=10, total_rows=100):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the first 100 rows of the CSV
    df = pd.read_csv(input_csv, nrows=total_rows)

    # Get column names
    columns = df.columns.tolist()

    # Calculate number of files
    num_files = total_rows // rows_per_file

    for i in range(num_files):
        # Calculate start and end indices for slicing
        start_idx = i * rows_per_file
        end_idx = start_idx + rows_per_file

        # Slice the dataframe
        chunk = df.iloc[start_idx:end_idx]

        # Define output file names
        csv_output = os.path.join(output_dir, f'chunk_{i + 1}.csv')
        pdf_output = os.path.join(output_dir, f'chunk_{i + 1}.pdf')

        # Save to CSV
        chunk.to_csv(csv_output, index=False)

        # Create PDF
        doc = fitz.open()
        page = doc.new_page()

        # Set starting position and font
        p = fitz.Point(50, 50)  # Starting point on page
        page.insert_text(p, f"Chunk {i + 1}", fontsize=14, fontname="helv", fontfile=None)
        p.y += 20

        # Add column headers
        header_text = ", ".join(columns)
        page.insert_text(p, header_text, fontsize=12, fontname="helv", fontfile=None)
        p.y += 20

        # Add data rows
        for _, row in chunk.iterrows():
            row_text = ", ".join(str(val) for val in row)
            page.insert_text(p, row_text, fontsize=10, fontname="helv", fontfile=None)
            p.y += 15

        # Save PDF
        doc.save(pdf_output)
        doc.close()


if __name__ == "__main__":
    # Example usage
    input_csv_file = "/home/yeganeh/PycharmProjects/Inr_task/data/invoice-dataset/invoices.csv"  # Replace with your CSV file path
    split_csv_to_files(input_csv_file)