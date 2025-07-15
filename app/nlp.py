# app/nlp.py
import difflib
from spacy.matcher import PhraseMatcher

def parse_natural_language(text, conn, tbl_name, nlp_model):
    """
    Uses spaCy to parse natural language and build a SQL query in parts.
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
    doc = nlp_model(corrected_text)

    lemmas = {token.lemma_ for token in doc}

    try:
        schema_df = conn.execute(f"DESCRIBE {tbl_name};").fetchdf()
        column_names = [col.lower() for col in schema_df['column_name'].tolist()]
    except Exception:
        column_names = []

    matcher = PhraseMatcher(nlp_model.vocab, attr="LOWER")
    patterns = [nlp_model.make_doc(col) for col in column_names]
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
