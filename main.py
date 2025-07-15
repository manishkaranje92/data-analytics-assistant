# main.py
import os
import duckdb
import pandas as pd
from flask import Flask, request, render_template, jsonify
import re
import difflib
import random
from itertools import combinations

# --- spaCy NLP Setup ---
import spacy
from spacy.matcher import PhraseMatcher

# Load a small English model. This will be downloaded on the first run.
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy NLP model loaded successfully.")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# In-memory storage for multiple tables
db_connection = None
loaded_tables = [] # Will store {'original_name': 'file.csv', 'table_name': 'file', 'schema': {}}

@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles multiple file uploads and loads them into DuckDB."""
    global db_connection, loaded_tables
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400
    if len(files) > 3:
        return jsonify({"error": "You can upload a maximum of 3 files."}), 400

    # Reset on new upload
    db_connection = duckdb.connect(database=':memory:', read_only=False)
    loaded_tables = []
    
    for file in files:
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                base_name = os.path.splitext(os.path.basename(filename))[0]
                table_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name)[:30]

                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(filepath)
                elif filename.lower().endswith('.parquet'):
                    df = pd.read_parquet(filepath)
                else:
                    continue # Skip unsupported files

                db_connection.register(f'df_{table_name}', df)
                db_connection.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df_{table_name};")

                schema_df = db_connection.execute(f"DESCRIBE {table_name}").fetchdf()
                loaded_tables.append({
                    'original_name': filename,
                    'table_name': table_name,
                    'schema': schema_df.to_dict(orient='records')
                })

            except Exception as e:
                return jsonify({"error": f"Error processing file {filename}: {str(e)}"}), 500
    
    if not loaded_tables:
        return jsonify({"error": "No valid files were processed."}), 400

    return jsonify({"message": f"Successfully loaded {len(loaded_tables)} files.", "tables": loaded_tables})

@app.route('/get_schema/<table_name>')
def get_schema(table_name):
    """Returns the schema for a specific table."""
    if not db_connection: return jsonify({"error": "No data loaded"}), 400
    try:
        schema_df = db_connection.execute(f"DESCRIBE {table_name}").fetchdf()
        return jsonify(schema_df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": f"Could not retrieve schema: {str(e)}"}), 500

@app.route('/get_data_quality/<table_name>')
def get_data_quality(table_name):
    """Performs a comprehensive data quality check on a specific table."""
    if not db_connection: return jsonify({"error": "No data loaded"}), 400
    try:
        schema_df = db_connection.execute(f"DESCRIBE {table_name}").fetchdf()
        columns = schema_df['column_name'].tolist()
        numeric_cols = [r['column_name'] for i, r in schema_df.iterrows() if 'INT' in r['column_type'] or 'FLOAT' in r['column_type'] or 'DECIMAL' in r['column_type']]
        categorical_cols = [r['column_name'] for i, r in schema_df.iterrows() if 'VARCHAR' in r['column_type'] or 'TEXT' in r['column_type']]

        total_rows = db_connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        distinct_rows = db_connection.execute(f"SELECT COUNT(*) FROM (SELECT DISTINCT * FROM {table_name})").fetchone()[0]
        
        null_checks = [f"SUM(CASE WHEN \"{col}\" IS NULL THEN 1 ELSE 0 END) AS \"{col}_nulls\"" for col in columns]
        null_counts = db_connection.execute(f"SELECT {', '.join(null_checks)} FROM {table_name}").fetchdf().to_dict(orient='records')[0]

        numerical_stats = {}
        for col in numeric_cols:
            try:
                stats = db_connection.execute(f'SELECT MIN("{col}"), MAX("{col}"), AVG("{col}"), STDDEV("{col}") FROM {table_name}').fetchone()
                numerical_stats[col] = {"min": f"{stats[0]:.2f}" if stats[0] is not None else "N/A", "max": f"{stats[1]:.2f}" if stats[1] is not None else "N/A", "avg": f"{stats[2]:.2f}" if stats[2] is not None else "N/A", "std_dev": f"{stats[3]:.2f}" if stats[3] is not None else "N/A"}
            except Exception: pass

        categorical_stats = {}
        for col in categorical_cols:
            try:
                distinct_count = db_connection.execute(f'SELECT COUNT(DISTINCT "{col}") FROM {table_name}').fetchone()[0]
                categorical_stats[col] = {"distinct_values": distinct_count}
            except Exception: pass

        quality_report = {
            "total_rows": total_rows,
            "duplicate_rows": total_rows - distinct_rows,
            "null_counts": null_counts,
            "numerical_stats": numerical_stats,
            "categorical_stats": categorical_stats
        }
        return jsonify(quality_report)
    except Exception as e:
        return jsonify({"error": f"Could not perform data quality check: {str(e)}"}), 500

@app.route('/get_suggestions')
def get_suggestions():
    """Analyzes schemas and suggests a balanced and diverse set of relevant queries."""
    global db_connection, loaded_tables
    if not loaded_tables: return jsonify({"error": "No data loaded"}), 400

    all_suggestions = []
    
    # Generate a pool of single-table suggestions
    for table_info in loaded_tables:
        table_name = table_info['table_name']
        schema = table_info['schema']
        numeric_cols = [c['column_name'] for c in schema if 'INT' in c['column_type'] or 'FLOAT' in c['column_type']]
        categorical_cols = [c['column_name'] for c in schema if 'VARCHAR' in c['column_type'] or 'TEXT' in c['column_type']]
        
        table_pool = []
        if categorical_cols:
            table_pool.append(f'SELECT "{random.choice(categorical_cols)}", COUNT(*) FROM {table_name} GROUP BY 1 ORDER BY 2 DESC LIMIT 10')
            table_pool.append(f'SELECT DISTINCT "{random.choice(categorical_cols)}" FROM {table_name} LIMIT 20')
        if numeric_cols:
            table_pool.append(f'SELECT MIN("{random.choice(numeric_cols)}"), MAX("{random.choice(numeric_cols)}"), AVG("{random.choice(numeric_cols)}") FROM {table_name}')
        if categorical_cols and numeric_cols:
            table_pool.append(f'SELECT "{random.choice(categorical_cols)}", SUM("{random.choice(numeric_cols)}") FROM {table_name} GROUP BY 1 ORDER BY 2 DESC LIMIT 10')
        
        all_suggestions.extend(random.sample(table_pool, min(len(table_pool), 2)))

    # Generate JOIN or UNION suggestions if multiple tables exist
    if len(loaded_tables) > 1:
        for t1_info, t2_info in combinations(loaded_tables, 2):
            t1_name = t1_info['table_name']
            t2_name = t2_info['table_name']
            
            # Create a canonical representation of schemas for comparison
            t1_schema_list = sorted([(c['column_name'].lower(), c['column_type']) for c in t1_info['schema']])
            t2_schema_list = sorted([(c['column_name'].lower(), c['column_type']) for c in t2_info['schema']])

            # If schemas are identical, suggest UNION
            if t1_schema_list == t2_schema_list:
                all_suggestions.append(f'SELECT * FROM {t1_name} UNION ALL SELECT * FROM {t2_name}')
            else:
                # Otherwise, check for common columns to suggest JOIN
                t1_cols = {c[0] for c in t1_schema_list}
                t2_cols = {c[0] for c in t2_schema_list}
                common_cols = t1_cols.intersection(t2_cols)
                
                if common_cols:
                    join_col = random.choice(list(common_cols))
                    all_suggestions.append(f'SELECT * FROM {t1_name} JOIN {t2_name} ON {t1_name}."{join_col}" = {t2_name}."{join_col}" LIMIT 10')

    unique_suggestions = list(set(all_suggestions))
    return jsonify(random.sample(unique_suggestions, min(3, len(unique_suggestions))))


@app.route('/chat', methods=['POST'])
def chat():
    """Processes user chat messages using the spaCy NLP library to generate and execute SQL."""
    global db_connection, loaded_tables
    user_message = request.json.get('message', '')
    if not db_connection: return jsonify({"response": "Please upload a data file first."})

    try:
        target_table = loaded_tables[0]['table_name'] if loaded_tables else None
        if not target_table: return jsonify({"response": "No valid tables loaded."})

        sql_query = parse_natural_language(user_message, db_connection, target_table)
        if not sql_query: return jsonify({"response": "Sorry, I couldn't understand that request."})
        
        result = db_connection.execute(sql_query).fetchdf()
        response_html = result.to_html(classes='table table-striped table-bordered', index=False, max_rows=20) if not result.empty else "Query executed, but it returned no results."
        return jsonify({"response": response_html})
    except Exception as e:
        return jsonify({"response": f"An error occurred: {str(e)}"})

@app.route('/sql_query', methods=['POST'])
def sql_query():
    """Executes a direct SQL query from the user."""
    global db_connection
    query = request.json.get('query', '')
    if not db_connection: return jsonify({"response": "Please upload a data file first."})
    try:
        result = db_connection.execute(query).fetchdf()
        # FIX: Improved message for empty results
        response_html = result.to_html(classes='table table-striped table-bordered', index=False, max_rows=50) if not result.empty else "Query executed successfully, but returned no matching rows."
        return jsonify({"response": response_html})
    except Exception as e:
        return jsonify({"response": f"<div class='alert alert-danger'><b>SQL Error:</b><br>{str(e)}</div>"})

def parse_natural_language(text, conn, tbl_name):
    """
    Uses spaCy to parse natural language and build a SQL query in parts.
    Handles combinations of keywords and corrects minor spelling mistakes.
    """
    KNOWN_KEYWORDS = [
        'distinct', 'order', 'by', 'group', 'describe', 'schema', 'structure', 'info',
        'column', 'field', 'header', 'count', 'row', 'record', 'entry', 'show', 'display',
        'select', 'get', 'find', 'list', 'give', 'average', 'avg', 'mean', 'sum', 'total',
        'min', 'minimum', 'max', 'maximum', 'desc', 'descending', 'asc', 'ascending'
    ]
    
    corrected_words = []
    for word in text.lower().split():
        matches = difflib.get_close_matches(word, KNOWN_KEYWORDS, n=1, cutoff=0.8)
        if matches:
            corrected_words.append(matches[0])
        else:
            corrected_words.append(word)
    
    corrected_text = " ".join(corrected_words)
    doc = nlp(corrected_text)

    lemmas = {token.lemma_ for token in doc}

    try:
        schema_df = conn.execute(f"DESCRIBE {tbl_name};").fetchdf()
        column_names = [col.lower() for col in schema_df['column_name'].tolist()]
    except Exception:
        column_names = []

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(col) for col in column_names]
    if patterns:
        matcher.add("ColumnMatcher", patterns)
    
    matches = matcher(doc)
    matched_columns = sorted(list(set([doc[start:end].text for _, start, end in matches])), key=lambda x: corrected_text.lower().find(x))

    is_distinct = 'distinct' in lemmas
    has_order_by = 'order' in lemmas and 'by' in lemmas
    has_group_by = 'group' in lemmas and 'by' in lemmas
    agg_map = {'average': 'AVG', 'avg': 'AVG', 'mean': 'AVG', 'sum': 'SUM', 'total': 'SUM', 'min': 'MIN', 'minimum': 'MIN', 'max': 'MAX', 'maximum': 'MAX', 'count': 'COUNT'}
    agg_lemmas = [lemma for lemma in lemmas if lemma in agg_map]

    if {'describe', 'schema', 'structure', 'info'}.intersection(lemmas):
        return f"DESCRIBE {tbl_name};"
    if {'column', 'field', 'header'}.intersection(lemmas):
        return f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{tbl_name}';"
    if 'count' in lemmas and any(l in lemmas for l in ['row', 'record', 'entry']) and not matched_columns:
        return f"SELECT COUNT(*) as total_rows FROM {tbl_name};"

    select_clause, group_by_clause, order_by_clause, limit_clause = "*", "", "", ""

    if has_order_by and matched_columns:
        try:
            order_by_keyword_index = corrected_text.lower().rfind('order by')
            order_by_text = corrected_text[order_by_keyword_index + 8:]
            order_col = next((c for c in matched_columns if c in order_by_text.lower()), matched_columns[-1])
            direction = "DESC" if 'desc' in lemmas or 'descending' in lemmas else "ASC"
            order_by_clause = f'ORDER BY "{order_col}" {direction}'
        except (ValueError, StopIteration): pass

    if has_group_by and matched_columns:
        try:
            group_by_keyword_index = corrected_text.lower().rfind('group by')
            group_by_text = corrected_text[group_by_keyword_index + 8:]
            group_col = next((c for c in matched_columns if c in group_by_text.lower()), matched_columns[0])
            group_by_clause = f'GROUP BY "{group_col}"'
            
            agg_func, agg_col = "COUNT", "*"
            if agg_lemmas:
                agg_func = agg_map[agg_lemmas[0]]
                agg_col_candidate = next((c for c in matched_columns if c != group_col), None)
                if agg_col_candidate:
                    agg_col = f'"{agg_col_candidate}"'
            
            select_clause = f'"{group_col}", {agg_func}({agg_col})'
        except (ValueError, StopIteration): pass
    
    elif matched_columns:
        select_cols = [f'"{c}"' for c in matched_columns]
        if agg_lemmas:
            agg_func = agg_map[agg_lemmas[0]]
            select_clause = f'COUNT(DISTINCT {select_cols[0]})' if agg_func == 'COUNT' and is_distinct else f'{agg_func}({select_cols[0]})'
        elif is_distinct:
            select_clause = f'DISTINCT {", ".join(select_cols)}'
        else:
            select_clause = ", ".join(select_cols)

    for ent in doc.ents:
        if ent.label_ == 'CARDINAL':
            try:
                limit_clause = f"LIMIT {int(ent.text)}"
                break
            except ValueError: continue

    if select_clause == "*" and not order_by_clause and not limit_clause:
        if any(l in lemmas for l in ['show', 'display', 'select', 'get', 'find', 'list', 'give']):
             return f"SELECT * FROM {tbl_name} LIMIT 10"
        else:
            return None

    query = f"SELECT {select_clause} FROM {tbl_name}"
    if group_by_clause: query += f" {group_by_clause}"
    if order_by_clause: query += f" {order_by_clause}"
    if limit_clause: query += f" {limit_clause}"

    return query

if __name__ == '__main__':
    app.run(debug=True, port=5001)
