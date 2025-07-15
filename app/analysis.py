# app/analysis.py
from flask import jsonify
import pandas as pd
import numpy as np

def get_schema_for_table(table_name, db_conn):
    """Returns the schema for a specific table."""
    if not db_conn: return jsonify({"error": "No data loaded"}), 400
    try:
        schema_df = db_conn.execute(f"DESCRIBE {table_name}").fetchdf()
        return jsonify(schema_df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": f"Could not retrieve schema: {str(e)}"}), 500

def get_data_quality_for_table(table_name, db_conn):
    """Performs a comprehensive data quality check on a specific table."""
    if not db_conn: return jsonify({"error": "No data loaded"}), 400
    try:
        schema_df = db_conn.execute(f"DESCRIBE {table_name}").fetchdf()
        columns = schema_df['column_name'].tolist()
        numeric_cols = [r['column_name'] for i, r in schema_df.iterrows() if 'INT' in r['column_type'] or 'FLOAT' in r['column_type'] or 'DECIMAL' in r['column_type']]
        categorical_cols = [r['column_name'] for i, r in schema_df.iterrows() if 'VARCHAR' in r['column_type'] or 'TEXT' in r['column_type']]

        total_rows = db_conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        distinct_rows = db_conn.execute(f"SELECT COUNT(*) FROM (SELECT DISTINCT * FROM {table_name})").fetchone()[0]
        
        null_checks = [f'SUM(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) AS "{col}_nulls"' for col in columns]
        null_counts = db_conn.execute(f"SELECT {', '.join(null_checks)} FROM {table_name}").fetchdf().to_dict(orient='records')[0]

        numerical_stats = {}
        for col in numeric_cols:
            try:
                stats = db_conn.execute(f'SELECT MIN("{col}"), MAX("{col}"), AVG("{col}"), STDDEV("{col}") FROM {table_name}').fetchone()
                numerical_stats[col] = {"min": f"{stats[0]:.2f}" if stats[0] is not None else "N/A", "max": f"{stats[1]:.2f}" if stats[1] is not None else "N/A", "avg": f"{stats[2]:.2f}" if stats[2] is not None else "N/A", "std_dev": f"{stats[3]:.2f}" if stats[3] is not None else "N/A"}
            except Exception: pass

        categorical_stats = {}
        for col in categorical_cols:
            try:
                distinct_count = db_conn.execute(f'SELECT COUNT(DISTINCT "{col}") FROM {table_name}').fetchone()[0]
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

def get_correlation_matrix(table_name, db_conn):
    """Calculates and returns the correlation matrix for numerical columns."""
    if not db_conn: return jsonify({"error": "No data loaded"}), 400
    try:
        schema_df = db_conn.execute(f"DESCRIBE {table_name}").fetchdf()
        numeric_cols = [r['column_name'] for i, r in schema_df.iterrows() if 'INT' in r['column_type'] or 'FLOAT' in r['column_type'] or 'DECIMAL' in r['column_type']]

        if len(numeric_cols) < 2:
            return jsonify({"error": "Not enough numerical columns to generate a correlation matrix."})

        quoted_numeric_cols = [f'"{c}"' for c in numeric_cols]
        select_query = f'SELECT {", ".join(quoted_numeric_cols)} FROM {table_name}'
        numerical_df = db_conn.execute(select_query).fetchdf()
        pandas_numeric_df = numerical_df.select_dtypes(include=np.number)

        if len(pandas_numeric_df.columns) < 2:
             return jsonify({"error": "Not enough valid numerical columns for correlation."})

        corr_df = pandas_numeric_df.corr()
        correlation_matrix = corr_df.fillna("N/A").to_dict(orient='index')
        
        return jsonify(correlation_matrix)

    except Exception as e:
        return jsonify({"error": f"Could not generate correlation matrix: {str(e)}"}), 500
