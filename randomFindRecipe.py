import pandas as pd
import random
import re

def preprocess_data(df):
    df['ingredients'] = df['ingredients'].fillna('').astype(str)
    df['instructions'] = df['instructions'].fillna('').astype(str)
    return df

def filter_recipes(df, time_range=None, keywords=None, top_k=3):
    # Lọc theo thời gian
    if time_range == '0-15':
        df = df[df['readyInMinutes'] <= 15]
    elif time_range == '16-30':
        df = df[(df['readyInMinutes'] > 15) & (df['readyInMinutes'] <= 30)]
    elif time_range == '>30':
        df = df[df['readyInMinutes'] > 30]

    # Lọc theo nguyên liệu
    if keywords:
        keywords = [keyword.strip().lower() for keyword in keywords.split(',')]  
        for keyword in keywords:
            df = df[df['ingredients'].str.lower().str.contains(keyword, na=False)] 

    if df.empty:
        return []

    return df.sample(n=min(top_k, len(df)), random_state=None).to_dict(orient='records')
