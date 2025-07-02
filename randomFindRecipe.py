import pandas as pd
import random
import re

def preprocess_data(df):
    df['ingredients'] = df['ingredients'].fillna('').astype(str)
    df['instructions'] = df['instructions'].fillna('').astype(str)
    return df

def filter_recipes(df, time_range=None, keywords=None, top_k=3):

    if time_range:
        # Kiểm tra xem có phải là thời gian cụ thể không (format: exact_X)
        if isinstance(time_range, int):
            specific_time = time_range
            
            # Tạo một cột đo lường độ chênh lệch thời gian
            df_with_diff = df.copy()
            df_with_diff['time_diff'] = abs(df_with_diff['readyInMinutes'] - specific_time)
            
            # Lọc các món có thời gian không quá chênh lệch 25% so với thời gian yêu cầu
            max_time_diff = max(5, int(specific_time * 0.25))  # Ít nhất 5 phút hoặc 25% thời gian
            df = df_with_diff[df_with_diff['time_diff'] <= max_time_diff].sort_values(by='time_diff')
        elif time_range == '0-15':
            df = df[df['readyInMinutes'] <= 15]
        elif time_range == '16-30':
            df = df[(df['readyInMinutes'] > 15) & (df['readyInMinutes'] <= 30)]
        elif time_range == '>30':
            df = df[df['readyInMinutes'] > 30]

    # Lọc theo nguyên liệu - Hỗ trợ nhiều nguyên liệu
    if keywords:
        if isinstance(keywords, str):
            keywords = [keyword.strip().lower() for keyword in keywords.split(',')]
        
        # Tạo bản sao để lưu trữ kết quả và thêm cột đếm
        filtered_df = df.copy()
        filtered_df['match_count'] = 0
        filtered_df['match_in_title'] = 0
        
        # Kiểm tra từng nguyên liệu
        for keyword in keywords:
            # Tạo pattern chính xác để tìm kiếm từ
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            
            # Kiểm tra trong ingredients
            mask_ingredients = filtered_df['ingredients'].str.lower().str.contains(pattern, na=False, regex=True)
            filtered_df.loc[mask_ingredients, 'match_count'] += 1
            
            # Kiểm tra trong title - cho điểm cao hơn
            mask_title = filtered_df['title'].str.lower().str.contains(pattern, na=False, regex=True)
            filtered_df.loc[mask_title, 'match_count'] += 1
            filtered_df.loc[mask_title, 'match_in_title'] += 1
        
        # Lọc ra các món có ít nhất 1 nguyên liệu khớp
        filtered_df = filtered_df[filtered_df['match_count'] > 0]
        
        # Ưu tiên các món có tất cả nguyên liệu (match_count >= len(keywords))
        all_ingredients = filtered_df[filtered_df['match_count'] >= len(keywords)]

        if not all_ingredients.empty:
            filtered_df = all_ingredients
        filtered_df = filtered_df.sort_values(by=['match_count', 'match_in_title'], ascending=False)
        df = filtered_df.drop(columns=['match_count', 'match_in_title'])
    if df.empty:
        return []
    if len(df) > top_k:
        return df.head(top_k).to_dict(orient='records')
    else:
        return df.to_dict(orient='records')