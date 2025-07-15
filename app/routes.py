# app/routes.py
from flask import render_template, request, jsonify
from app import app, db_connection, loaded_tables, nlp_model
from app.data_processing import process_uploaded_files
from app.analysis import get_schema_for_table, get_data_quality_for_table, get_correlation_matrix
from app.suggestions import get_query_suggestions
from app.nlp import parse_natural_language
from app.charting import generate_chart

@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handles multiple file uploads."""
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400
    if len(files) > 3:
        return jsonify({"error": "You can upload a maximum of 3 files."}), 400

    message, tables, error = process_uploaded_files(files, db_connection, app.config['UPLOAD_FOLDER'])
    if error:
        return jsonify({"error": error}), 500
    
    loaded_tables.clear()
    loaded_tables.extend(tables)
    
    return jsonify({"message": message, "tables": tables})

@app.route('/get_schema/<table_name>')
def get_schema(table_name):
    """Returns the schema for a specific table."""
    return get_schema_for_table(table_name, db_connection)

@app.route('/get_data_quality/<table_name>')
def get_data_quality(table_name):
    """Performs a comprehensive data quality check on a specific table."""
    return get_data_quality_for_table(table_name, db_connection)

@app.route('/get_correlation/<table_name>')
def get_correlation(table_name):
    """Generates a correlation matrix for a specific table."""
    return get_correlation_matrix(table_name, db_connection)

@app.route('/get_suggestions')
def get_suggestions():
    """Analyzes schemas and suggests relevant queries."""
    return get_query_suggestions(loaded_tables)

@app.route('/execute', methods=['POST'])
def execute_query():
    """Executes a query (SQL or NL) and returns an HTML table."""
    data = request.json
    query_text = data.get('query', '')
    query_type = data.get('type', 'sql')  # 'sql' or 'chat'

    if not db_connection:
        return jsonify({"table": "Please upload a data file first."}), 400

    sql_query = ""
    try:
        if query_type == 'chat':
            if not loaded_tables:
                return jsonify({"table": "Please upload a data file first."}), 400
            target_table = loaded_tables[0]['table_name']
            sql_query = parse_natural_language(query_text, db_connection, target_table, nlp_model)
            if not sql_query:
                return jsonify({"table": "Sorry, I couldn't understand that request."})
        else:  # query_type == 'sql'
            sql_query = query_text

        result_df = db_connection.execute(sql_query).fetchdf()
        response_html = result_df.to_html(classes='table table-striped table-bordered', index=False, max_rows=50) if not result_df.empty else "Query executed successfully, but returned no matching rows."
        return jsonify({"table": response_html})

    except Exception as e:
        error_html = f"<div class='alert alert-danger'><b>SQL Error:</b><br>{str(e)}</div>"
        return jsonify({"table": error_html}), 500

@app.route('/visualize', methods=['POST'])
def visualize_query():
    """Executes a query (SQL or NL) and returns a Plotly chart."""
    data = request.json
    query_text = data.get('query', '')
    query_type = data.get('type', 'sql')  # 'sql' or 'chat'

    if not db_connection:
        return jsonify({"chart": None, "error": "Please upload a data file first."}), 400

    sql_query = ""
    try:
        if query_type == 'chat':
            if not loaded_tables:
                return jsonify({"chart": None, "error": "Please upload a data file first."}), 400
            target_table = loaded_tables[0]['table_name']
            sql_query = parse_natural_language(query_text, db_connection, target_table, nlp_model)
            if not sql_query:
                return jsonify({"chart": None, "error": "Sorry, I couldn't understand that request."})
        else:  # query_type == 'sql'
            sql_query = query_text

        result_df = db_connection.execute(sql_query).fetchdf()
        if result_df.empty:
            return jsonify({"chart": None, "error": "Query returned no data to visualize."})

        chart_json = generate_chart(result_df)
        if not chart_json:
            return jsonify({"chart": None, "error": "Could not generate a suitable chart for this data."})

        return jsonify({"chart": chart_json})

    except Exception as e:
        return jsonify({"chart": None, "error": f"An error occurred: {str(e)}"}), 500
