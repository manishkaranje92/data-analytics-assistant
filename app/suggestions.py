# app/suggestions.py
from flask import jsonify
import random
from itertools import combinations

def get_query_suggestions(loaded_tables):
    """Analyzes schemas and suggests a balanced and diverse set of relevant queries."""
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
            
            t1_schema_list = sorted([(c['column_name'].lower(), c['column_type']) for c in t1_info['schema']])
            t2_schema_list = sorted([(c['column_name'].lower(), c['column_type']) for c in t2_info['schema']])

            if t1_schema_list == t2_schema_list:
                all_suggestions.append(f'SELECT * FROM {t1_name} UNION ALL SELECT * FROM {t2_name}')
            else:
                t1_cols = {c[0] for c in t1_schema_list}
                t2_cols = {c[0] for c in t2_schema_list}
                common_cols = t1_cols.intersection(t2_cols)
                
                if common_cols:
                    join_col = random.choice(list(common_cols))
                    all_suggestions.append(f'SELECT * FROM {t1_name} JOIN {t2_name} ON {t1_name}."{join_col}" = {t2_name}."{join_col}" LIMIT 10')

    unique_suggestions = list(set(all_suggestions))
    return jsonify(random.sample(unique_suggestions, min(3, len(unique_suggestions))))
