# app/charting.py
import plotly.express as px
import pandas as pd
import numpy as np

def convert_numpy_types(obj):
    """
    Recursively converts numpy and pandas types to native Python types
    for JSON serialization.
    """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj

def generate_chart(df: pd.DataFrame):
    """
    Analyzes a DataFrame and generates a suitable Plotly chart dictionary if appropriate.

    Args:
        df: The input pandas DataFrame from the user's query.

    Returns:
        A dictionary representing a Plotly figure, or None if no suitable
        chart can be generated.
    """
    if df.empty or len(df.columns) < 1:
        return None

    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Attempt to convert object columns that look numeric
    for col in df.select_dtypes(include=['object']).columns:
        try:
            # Use errors='coerce' to turn non-numeric values into NaT/NaN
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            # Only convert if the majority of values are numeric
            if numeric_col.notna().sum() / len(df.index) > 0.5:
                 df[col] = numeric_col
        except (ValueError, TypeError):
            pass # Ignore columns that can't be converted

    # Identify column types after potential conversion
    cols = df.columns
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Prioritize date/datetime columns if they exist
    date_cols = []
    for col in df.select_dtypes(include=['object']).columns:
        try:
            # Attempt to convert to datetime, coercing errors
            temp_col = pd.to_datetime(df[col], errors='coerce')
            # Check if a significant portion of the column was successfully converted
            if temp_col.notna().sum() / len(df.index) > 0.5:
                df[col] = temp_col
                date_cols.append(col)
        except Exception:
            continue
    
    # Also include columns that are already datetime type
    date_cols.extend(df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist())
    date_cols = list(set(date_cols)) # Get unique date columns

    if date_cols:
        df = df.sort_values(by=date_cols[0])


    fig = None

    # --- Chart Generation Heuristics ---

    # 1. Time Series Line Chart: One date column, one numeric column
    if len(date_cols) >= 1 and len(num_cols) >= 1:
        print(f"Date columns: {date_cols}, Numeric columns: {num_cols}")
        print("generating line chart")
        x_ax, y_ax = date_cols[0], num_cols[0]
        fig = px.line(df, x=x_ax, y=y_ax, title=f'{y_ax} over Time')

    # 2. Bar Chart: One categorical column, one numerical column
    elif len(cat_cols) == 1 and len(num_cols) == 1:
        print(f"Categorical columns: {cat_cols}, Numeric columns: {num_cols}")
        print("generating bar chart")
        cat_col, num_col = cat_cols[0], num_cols[0]
        # Limit categories to avoid overly cluttered charts
        df_to_plot = df.nlargest(25, num_col) if df[cat_col].nunique() > 25 else df
        fig = px.bar(df_to_plot, x=cat_col, y=num_col, title=f'{num_col} by {cat_col}')

    # 3. Scatter Plot: Two numerical columns
    elif len(num_cols) >= 2:
        print(f"Numerical columns: {num_cols}")
        print("generating scatter plot")
        x_ax, y_ax = num_cols[0], num_cols[1]
        title = f'{y_ax} vs. {x_ax}'
        # Add a third column for color if it's categorical and has low cardinality
        color_col = next((c for c in cat_cols if df[c].nunique() < 15), None)
        if color_col:
            title += f' by {color_col}'
        fig = px.scatter(df, x=x_ax, y=y_ax, title=title, color=color_col)
        
    # 4. Pie Chart / Bar Chart of Counts: One categorical column
    elif len(cat_cols) == 1 and len(num_cols) == 0:
        print(f"Categorical columns: {cat_cols}")
        print("generating pie/bar chart")
        cat_col = cat_cols[0]
        counts = df[cat_col].value_counts().reset_index()
        counts.columns = [cat_col, 'count']
        if 2 <= df[cat_col].nunique() <= 10:
            fig = px.pie(counts, names=cat_col, values='count', title=f'Distribution of {cat_col}')
        else:
            fig = px.bar(counts.head(25), x=cat_col, y='count', title=f'Top 25 Counts for {cat_col}')

    # 5. Histogram: One numerical column
    elif len(num_cols) == 1 and len(cat_cols) == 0:
        print(f"Numerical columns: {num_cols}")
        print("generating histogram")
        num_col = num_cols[0]
        fig = px.histogram(df, x=num_col, title=f'Distribution of {num_col}')

    if fig:
        print("Chart generated successfully")
        # Update layout for better appearance
        fig.update_layout(
            margin=dict(l=50, r=20, t=50, b=50),
            title_x=0.5,
            paper_bgcolor="white",
            plot_bgcolor="rgba(240,240,240,0.95)",
        )
        # Correctly call to_dict() on the figure object itself
        fig_dict = fig.to_dict()
        # print("genearated chart dict:", fig_dict)
        # Convert any numpy types to native Python types for JSON serialization
        return convert_numpy_types(fig_dict)

    return None # No suitable chart found
