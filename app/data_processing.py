# app/data_processing.py
import os
import re
import pandas as pd

def process_uploaded_files(files, db_conn, upload_folder):
    """
    Processes a list of uploaded files, loads them into DuckDB, and returns table info.
    """
    tables_info = []
    
    for file in files:
        filename = file.filename
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        
        try:
            base_name = os.path.splitext(os.path.basename(filename))[0]
            table_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name)[:30]

            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.lower().endswith('.parquet'):
                df = pd.read_parquet(filepath)
            else:
                continue

            db_conn.register(f'df_{table_name}', df)
            db_conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df_{table_name};")

            schema_df = db_conn.execute(f"DESCRIBE {table_name}").fetchdf()
            tables_info.append({
                'original_name': filename,
                'table_name': table_name,
                'schema': schema_df.to_dict(orient='records')
            })

        except Exception as e:
            return None, None, f"Error processing file {filename}: {str(e)}"
    
    if not tables_info:
        return None, None, "No valid files were processed."

    message = f"Successfully loaded {len(tables_info)} files."
    return message, tables_info, None
