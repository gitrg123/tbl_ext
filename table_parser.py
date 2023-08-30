import argparse, os, fitz, pandas as pd, torch
from src.inference import TableExtractionPipeline
from typing import List, Any
from PIL import Image

# Create inference pipeline
table_extraction_pipeline = TableExtractionPipeline(
    det_config_path='./src/inference/detection_config.json', 
    det_model_path='../pubtables1m_detection_detr_r18.pth', 
    det_device='cuda' if torch.cuda.is_available() else "cpu", 
    str_config_path='./src/inference/structure_config.json', 
    str_model_path='../pubtables1m_structure_detr_r18.pth', 
    str_device='cuda' if torch.cuda.is_available() else "cpu"
)

def get_extracted_data(pil_image):
    # Recognize table(s) from image
    tokens = []
    extracted_tables = table_extraction_pipeline.recognize(
        pil_image, tokens, out_objects=True, out_cells=True, out_html=True, out_csv=True)

    # # Select table (there could be more than one)
    # extracted_table = extracted_tables[0]

    # # Get output in desired format
    # objects = extracted_table['objects']
    # cells = extracted_table['cells']
    # csv = extracted_table['csv']
    # html = extracted_table['html']
    return extracted_tables

def read_PDF_and_convert_pages_into_images(pdf_filepath) -> List[Any]:        
    pdf_document = fitz.open(pdf_filepath)
    pdf_page_images = []
    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # Adjust resolution as needed
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pdf_page_images.append(image) 
        # image.save(f"{pdf_filepath.rstrip(".pdf")}_{page_number}.png")
    pdf_document.close()
    return pdf_page_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument("--input_pdf_filepath", required=True, help="path to input PDF file")  
    parser.add_argument("--output_folder_path", required=True, help="path to output folder for saving parsed data")  
    args = parser.parse_args() 
    
    pdf_filepath = str(args.input_pdf_filepath)
    output_folder_path = str(args.output_folder_path)
    os.makedirs(output_folder_path, exist_ok=True)

    page_images = read_PDF_and_convert_pages_into_images(pdf_filepath)
    tables_book = []
    for page_num, page_image in enumerate(page_images):
        extracted_tables = get_extracted_data(page_image)
        for table_num, table_data in enumerate(extracted_tables):
            dataframe = pd.read_csv(table_data["csv"])
            markdown_table = dataframe.to_markdown(index=False)
            markdown_table = dataframe.to_json(orient="records")

            tables_book.append({
                "page_number": page_num,
                "table_number": table_num,
                "markdown_table": markdown_table,
                "json_table": markdown_table
            })
    
    tables_book = pd.DataFrame(tables_book)
    tables_filename = os.path.basename(pdf_filepath.rstrip(".pdf") + ".json")
    tables_book.to_json(os.path.join(output_folder_path, tables_filename), orient='records', indent=1)