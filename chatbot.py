from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import os
import pickle
import unicodedata
import numpy as np
from recommender import clean_html
from model_trainer import load_all_models
from chatbot_recommender import ChatbotRecommender

app = Flask(__name__, template_folder='templates', static_folder='static')
df, similarity_matrices = load_all_models()

recommender = ChatbotRecommender(df, similarity_matrices)

models = [
    {"id": "doc2vec", "name": "Doc2Vec", "description": "Sử dụng mô hình Doc2Vec để tìm món ăn tương tự"},
    {"id": "tfidf", "name": "TF-IDF", "description": "Dùng TF-IDF để đánh giá sự tương đồng giữa các món ăn"},
    {"id": "pyvi", "name": "PyVi", "description": "Sử dụng thư viện PyVi để xử lý tiếng Việt và tìm món tương tự"},
    {"id": "bm25", "name": "BM25", "description": "Áp dụng thuật toán BM25 cho việc tìm món ăn tương tự"},
    {"id": "vncorenlp", "name": "VnCoreNLP", "description": "Dùng VnCoreNLP để xử lý tiếng Việt chuyên sâu và tìm món tương tự"}
]

cooking_times = [
    {"id": "0-15", "name": "0-15 phút", "description": "Món ăn nhanh, dưới 15 phút"},
    {"id": "16-30", "name": "16-30 phút", "description": "Món ăn có thời gian vừa phải, từ 16-30 phút"},
    {"id": ">30", "name": "Trên 30 phút", "description": "Món ăn cần nhiều thời gian chuẩn bị, trên 30 phút"}
]

greetings = [
    "Xin chào! Tôi là chatbot gợi ý món ăn. Tôi có thể giúp bạn tìm công thức nấu ăn yêu thích hoặc gợi ý món mới. Bạn muốn ăn gì hôm nay?",
    "Chào bạn! Hôm nay bạn muốn nấu món gì?",
    "Rất vui được gặp bạn! Tôi có thể gợi ý món ăn cho bạn. Bạn đang tìm kiếm món gì?",
    "Chào mừng đến với trợ lý món ăn! Tôi có thể giúp gì cho bạn?",
    "Xin chào! Đói bụng rồi phải không? Tôi sẽ giúp bạn tìm món ngon!",
    "Muốn ăn gì nào bro?"
]

# Danh sách nguyên liệu phổ biến được cập nhật từ phân tích 1,841 công thức trong recipes.csv
# Sắp xếp theo tần suất xuất hiện từ cao đến thấp

popular_ingredients = [
    # Top 50 nguyên liệu phổ biến nhất
    'dầu ăn', 'đường', 'tiêu', 'muối', 'nước mắm', 'tỏi băm', 'hành lá', 'ngò rí', 'tương ớt', 'cà rốt',
    'hành tỏi băm', 'hành tím băm', 'bột năng', 'hành tây', 'ớt băm', 'dầu điều', 'nước cốt chanh', 'dầu mè', 'rượu trắng', 'giò sống',
    'trứng gà', 'ớt sừng', 'ngò gai', 'tỏi', 'dầu hào', 'tôm sú', 'xà lách', 'ớt hiểm', 'cà chua', 'sả băm',
    'cà chua bi', 'dưa leo', 'bột mì', 'tương cà', 'nước dừa tươi', 'sả', 'nước tương', 'rau răm', 'nấm rơm', 'mè trắng rang',
    'mè rang', 'hành tím', 'chanh', 'rau om', 'hành phi', 'mật ong', 'đậu phộng rang', 'gừng', 'ớt bột', 'ớt',
    
    # Top 51-100
    'nấm kim châm', 'cần tây', 'hành boaro', 'thịt ba chỉ', 'giá', 'gừng băm', 'ngũ vị hương', 'húng lủi', 'tỏi phi', 'nước dùng',
    'bột nghệ', 'bơ lạt', 'thơm', 'rau thơm', 'nước cốt dừa', 'sườn non', 'ớt sừng băm', 'thì là', 'nấm mèo', 'ớt sừng cắt sợi',
    'đậu hũ non', 'sa tế', 'đậu hũ trắng', 'bông cải xanh', 'bột xù', 'đậu hũ chiên', 'thịt bò phi lê', 'húng quế', 'khoai tây', 'tiêu xay',
    'tôm khô', 'ớt hiểm băm', 'nấm bào ngư', 'cần tàu', 'hành ngò', 'tỏi ớt băm', 'bún', 'bắp non', 'bông hẹ', 'nấm đùi gà',
    'củ cải trắng', 'tôm đất', 'tương xí muội', 'củ sắn', 'bột bắp', 'gừng cắt sợi', 'nấm hương', 'thịt nạc dăm', 'cải thảo', 'sữa tươi không đường',
    
    # Top 101-150
    'bún tươi', 'thịt ba rọi', 'rau mầm', 'hành tím phi', 'đầu hành lá', 'dầu màu điều', 'bắp cải', 'dầu hào chay', 'thịt xay', 'tôm tươi',
    'bí đỏ', 'hành lá cắt nhỏ', 'sữa đặc', 'thịt nạc dăm xay', 'mè trắng', 'nước dùng gà', 'nấm rơm búp', 'bột cà ri', 'mực ống', 'củ sen',
    'mù tạt vàng', 'thịt cua', 'rau sống', 'đường phèn', 'cá thác lác', 'ớt chuông xanh', 'bơ', 'củ năng', 'rau húng lủi', 'đồ chua',
    'thịt ức gà', 'tiêu xanh', 'nước bột năng', 'cánh gà', 'mực lá', 'ớt cắt lát', 'muối hột', 'xốt mayonnaise', 'nấm đông cô tươi', 'bột ngọt',
    'bánh tráng', 'nước hành tỏi', 'nấm linh chi', 'bắp cải tím', 'rau nêm ngò rí', 'ớt sợi', 'nghêu', 'dừa tươi', 'trứng cút', 'lá lốt',
    
    # Top 151-200
    'đậu hà lan', 'nước dừa', 'giá sống', 'sữa tươi', 'ngò tây', 'nước cốt me', 'ớt bột paprika', 'bột gạo', 'nước tỏi', 'ớt chuông đỏ',
    'hẹ lá', 'rau quế', 'hành tím bào', 'rau nhút', 'tôm sú lột vỏ', 'nước me', 'chả lụa chay', 'khoai lang', 'bắp hạt', 'dừa xiêm',
    'tía tô', 'thịt heo xay', 'hành', 'cà phê', 'cần tây lá', 'bông cải trắng', 'bánh phở', 'cá bơn', 'cá thu', 'nghệ tươi',
    'bắp mỹ', 'rau cần', 'nấm linh chi tươi', 'bánh mì', 'thịt nạc vai', 'tỏi tây', 'rau má', 'cà rốt baby', 'nấm đùi gà tươi', 'rau dền',
    'khổ qua', 'cà pháo', 'me', 'đậu phụ', 'nước cam', 'thịt sườn', 'cua đồng', 'tôm he', 'cà tim', 'nấm hải sản',
    'rau cải', 'lạc rang', 'bánh đa', 'măng tây', 'cá hồi', 'cà chua xanh', 'thịt gà', 'bắp cải con', 'đậu xanh', 'nấm mỡ'
]

class ConversationManager:
    def __init__(self):
        self.sessions = {}
    
    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'state': 'initial',
                'context': {
                    'ingredients': [],
                    'time_range': None,
                    'last_ingredients': [], 
                    'last_query': ''
                },
                'last_recipes': [],
            }
        return self.sessions[session_id]
    
    def update_session(self, session_id, state=None, context=None, last_recipes=None):
        session = self.get_session(session_id)
        if state is not None:
            session['state'] = state
        if context is not None:
            session['context'].update(context)
        if last_recipes is not None:
            session['last_recipes'] = last_recipes
        return session

conversation_manager = ConversationManager()

def preprocess_text(text):
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = re.sub(r'[!.,?;:()]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_time_range(text):
    text = text.lower()
    specific_time_match = re.search(r'(\d+)\s*(phút|p|mins?|minutes?)', text)
    if specific_time_match:
        time_value = int(specific_time_match.group(1))
        return time_value
    if re.search(r'(0-15|dưới 15|ít hơn 15|nhanh|dưới 15 phút|ít thời gian)', text):
        return '0-15'
    elif re.search(r'(16-30|từ 16|khoảng 20|vừa phải|16-30 phút|thời gian vừa)', text):
        return '16-30'
    elif re.search(r'(>30|trên 30|nhiều thời gian|thời gian dài|lâu|trên 30 phút|chậm)', text):
        return '>30'
    return None

def get_time_display(time_range):
    if time_range is None:
        return ""
    if isinstance(time_range, int):
        return f"trong khoảng {time_range} phút"
    elif time_range == '0-15':
        return "trong khoảng 0-15 phút"
    elif time_range == '16-30':
        return "trong khoảng 16-30 phút"
    elif time_range == '>30':
        return "trong hơn 30 phút"
    return f"trong khoảng thời gian {time_range}"

def extract_ingredients(text):
    text = text.lower()
    found_ingredients = []
    #  Tìm kiếm chính xác các nguyên liệu trong văn bản
    for ingredient in popular_ingredients:
        # Sử dụng regex để tìm từ hoàn chỉnh, tránh matches một phần
        pattern = r'\b' + re.escape(ingredient) + r'\b'
        if re.search(pattern, text):
            found_ingredients.append(ingredient)
    
    #  Nếu chưa tìm thấy gì, thử tìm kiếm từ con (substring matching)
    if not found_ingredients:
        # Tạo mapping từ khóa đơn giản -> nguyên liệu phức tạp
        simple_keywords = {}
        for ingredient in popular_ingredients:
            words = ingredient.split()
            # Lấy từ cuối cùng làm keyword chính (ví dụ: "thịt gà" -> "gà")
            if len(words) > 1:
                main_keyword = words[-1]
                if main_keyword not in simple_keywords:
                    simple_keywords[main_keyword] = []
                simple_keywords[main_keyword].append(ingredient)
            else:
                # Nếu là từ đơn, thêm vào luôn
                simple_keywords[ingredient] = [ingredient]
        
        # Tìm kiếm dựa trên từ khóa đơn giản
        text_words = text.split()
        for word in text_words:
            word = word.strip('.,!?;:()[]{}')  # Loại bỏ dấu câu
            if word in simple_keywords:
                # Ưu tiên nguyên liệu đơn giản trước (ít từ hơn)
                matching_ingredients = sorted(simple_keywords[word], key=len)
                found_ingredients.extend(matching_ingredients[:1])  # Chỉ lấy 1 cái đầu tiên
    
    #  Tìm các từ nguyên liệu được nối bằng từ khóa liên kết
    ingredient_list_indicators = ['và', 'cùng với', 'kết hợp', 'kèm theo', 'cùng', 'với']
    extra_ingredients = []
    
    for indicator in ingredient_list_indicators:
        if indicator in text:
            # Phân tích cú pháp "A và B"
            parts = re.split(r'\s*' + re.escape(indicator) + r'\s*', text)
            if len(parts) > 1:
                for part in parts[1:]:
                    # Lấy từ đầu tiên sau từ khóa
                    potential_ingredient = part.strip().split()[0] if part.strip() else ""
                    if potential_ingredient and potential_ingredient not in found_ingredients:
                        
                        # Kiểm tra xem có phải là nguyên liệu hợp lệ không
                        # Trước tiên kiểm tra trực tiếp
                        if potential_ingredient in popular_ingredients:
                            extra_ingredients.append(potential_ingredient)
                        else:
                            # Tìm kiếm nguyên liệu chứa từ này
                            for ingredient in popular_ingredients:
                                if potential_ingredient in ingredient.split():
                                    extra_ingredients.append(ingredient)
                                    break
    
    # Thêm các nguyên liệu phụ đã tìm thấy
    found_ingredients.extend(extra_ingredients)
    
    #  Tìm các nguyên liệu ngăn cách bởi dấu phẩy
    comma_separated = re.split(r'\s*,\s*', text)
    if len(comma_separated) > 1:
        for item in comma_separated:
            item = item.strip()
            if item and item not in found_ingredients and len(item.split()) <= 2:
                # Kiểm tra xem có phải nguyên liệu hợp lệ không
                if item in popular_ingredients:
                    found_ingredients.append(item)
                else:
                    # Tìm nguyên liệu chứa từ này
                    for ingredient in popular_ingredients:
                        if item in ingredient:
                            found_ingredients.append(ingredient)
                            break
    
    #  Loại bỏ trùng lặp và sắp xếp theo độ ưu tiên
    found_ingredients = list(dict.fromkeys(found_ingredients))  # Loại bỏ trùng lặp nhưng giữ thứ tự
    
    # Ưu tiên các nguyên liệu đơn giản hơn (ít từ hơn)
    found_ingredients.sort(key=lambda x: (len(x.split()), x))
    
    return found_ingredients

def extract_model_type(text):
    text = text.lower()
    for model in models:
        if model['id'].lower() in text or model['name'].lower() in text:
            return model['id']
    return None

def identify_intent(text, session):
    text = preprocess_text(text)
    
    if re.search(r'(xin chào|chào|hello|hi|hey|alo)', text):
        return 'greeting'
    
    if re.search(r'(mô hình|model|phương pháp|cách đề xuất|thuật toán)', text):
        return 'ask_models'
    
    if re.search(r'(ngẫu nhiên|random|bất kỳ|gợi ý món|đề xuất món|món gì|món nào)', text):
        ingredients = extract_ingredients(text)
        if ingredients:
            return 'filter_recipe'  
        else:
            return 'popular_recipes'
    
    if re.search(r'(tương tự|giống|như|món giống|món tương tự)', text):
        return 'similar_recipe'
    
    if re.search(r'(phút|thời gian|nhanh|nguyên liệu|làm món|nấu món)', text) or extract_ingredients(text):
        return 'filter_recipe'
    
    if re.search(r'(tìm món|tìm kiếm|món|cách nấu|cách làm|làm sao)', text):
        return 'search_recipe'
    
    if re.search(r'(đánh giá|review|nhận xét|comment|feedback)', text):
        return 'rate_recipe'
    
    if re.search(r'(giúp|help|hướng dẫn|instruction|cách dùng)', text):
        return 'help'
    
    if session['state'] == 'ask_ingredients':
        return 'provide_ingredients'
    
    if session['state'] == 'ask_recipe_name':
        return 'provide_recipe_name'
    
    ingredients = extract_ingredients(text)
    if ingredients:
        return 'filter_recipe'
    
    return 'general_query'

def clean_recipe_data(recipes):
    if not recipes or len(recipes) == 0:
        print("Không tìm thấy món ăn phù hợp!")
        return []
    
    print(f"Tìm thấy {len(recipes)} món ăn")
    cleaned_recipes = []

    for recipe in recipes:
        if not isinstance(recipe, dict):
            print(f"Lỗi: recipe không phải là dict: {recipe}")
            continue
        
        recipe['instructions'] = clean_html(recipe.get('instructions', ''))
        recipe['ingredients'] = clean_html(recipe.get('ingredients', ''))
        
        required_fields = ['title', 'ingredients', 'instructions', 'readyInMinutes']
        for field in required_fields:
            if field not in recipe or not recipe[field]:
                if field == 'readyInMinutes':
                    recipe[field] = 30  
                else:
                    recipe[field] = f"Không có thông tin {field}"
        
        recipe['full_instructions'] = recipe['instructions']
        recipe['full_ingredients'] = recipe['ingredients']
        
        if len(recipe.get('instructions', '')) > 500:
            recipe['instructions'] = recipe['instructions'][:500] + '...'
        
        if len(recipe.get('ingredients', '')) > 300:
            recipe['ingredients'] = recipe['ingredients'][:300] + '...'
        
        cleaned_recipes.append(recipe)
    
    return cleaned_recipes

def process_message(message, session_id='default'):
    session = conversation_manager.get_session(session_id)
    session['context']['last_query'] = message
    intent = identify_intent(message, session)
    print(f"Ý định: {intent}")
    
    current_ingredients = extract_ingredients(message)
    
    if current_ingredients:
        session['context']['last_ingredients'] = current_ingredients
    
    if intent == 'greeting':
        conversation_manager.update_session(
            session_id, 
            state='initial', 
            context={
                'ingredients': [],
                'time_range': None,
                'last_ingredients': [],
                'last_query': message
            }
        )
        return {
            "type": "greeting",
            "message": "Xin chào! Tôi là chatbot gợi ý món ăn thông minh. Tôi có thể giúp bạn tìm công thức món ăn yêu thích hoặc gợi ý món dựa trên nhiều nguyên liệu bạn có. Bạn có thể hỏi tôi về món có nguyên liệu A và B, hoặc món ăn có nhiều nguyên liệu khác nhau."
        }
    
    elif intent == 'ask_models':
        return {
            "type": "models",
            "message": "Tôi có các mô hình gợi ý món ăn sau đây:",
            "models": models
        }
    
    elif intent == 'popular_recipes':
        popular_recipes = recommender.get_popular_recipes(5)
        clean_results = clean_recipe_data(popular_recipes)
        conversation_manager.update_session(session_id, state='popular', last_recipes=clean_results)
        return {
            "type": "search_results",
            "message": "Đây là một số món ăn phổ biến bạn có thể thử:",
            "recipes": clean_results
        }
    
    elif intent == 'similar_recipe':
        possible_titles = [title for title in df['title'].tolist() if title.lower() in message.lower()]
        
        if possible_titles:
            model_name = extract_model_type(message) or 'doc2vec'
            recipe_title = possible_titles[0]
            recommendations = recommender.get_similar_recipes(recipe_title, model_name)
            clean_recommendations = clean_recipe_data(recommendations)
            
            conversation_manager.update_session(
                session_id, 
                state='recommended', 
                context={'last_recipe': recipe_title, 'model': model_name},
                last_recipes=clean_recommendations
            )
            
            return {
                "type": "similar_recipes",
                "message": f"Dưới đây là các món ăn tương tự với {recipe_title}:",
                "recipes": clean_recommendations,
                "model": model_name
            }
        else:
            conversation_manager.update_session(session_id, state='ask_recipe_name')
            return {
                "type": "ask_recipe",
                "message": "Bạn muốn tìm món tương tự với món nào? Hãy cho tôi biết tên món ăn."
            }
    
    elif intent == 'provide_recipe_name' and session['state'] == 'ask_recipe_name':
        recipes = recommender.search_recipes_by_name(message)
        
        if recipes:
            recipe_title = recipes[0]['title']
            model_name = session['context'].get('model', 'doc2vec')
            
            recommendations = recommender.get_similar_recipes(recipe_title, model_name)
            clean_recommendations = clean_recipe_data(recommendations)
            
            conversation_manager.update_session(
                session_id, 
                state='recommended', 
                context={'last_recipe': recipe_title, 'model': model_name},
                last_recipes=clean_recommendations
            )
            
            return {
                "type": "similar_recipes",
                "message": f"Dưới đây là các món ăn tương tự với {recipe_title}:",
                "recipes": clean_recommendations,
                "model": model_name
            }
        else:
            return {
                "type": "not_found",
                "message": "Tôi không tìm thấy món ăn nào với tên như vậy. Bạn có thể thử một tên món ăn khác hoặc yêu cầu gợi ý món theo nguyên liệu."
            }
    
    elif intent == 'filter_recipe':
        ingredients = current_ingredients
        
        if not ingredients and session['context'].get('last_ingredients'):
            ingredients = session['context']['last_ingredients']
        
        time_range = extract_time_range(message) or session['context'].get('time_range')
        
        if ingredients:
            results = recommender.filter_recipes_by_time_and_ingredients(time_range, ingredients)
            clean_results = clean_recipe_data(results)
            
            conversation_manager.update_session(
                session_id, 
                state='filtered', 
                context={
                    'time_range': time_range, 
                    'ingredients': ingredients,
                    'last_ingredients': ingredients
                },
                last_recipes=clean_results
            )
            
            ingredient_str = ', '.join(ingredients)
            time_str = ""
            if time_range:
                time_str = f" trong khoảng thời gian {time_range} phút"
            
            if clean_results:
                all_ingredients_found = True
                if len(ingredients) > 1:
                    for recipe in clean_results:
                        for ingredient in ingredients:
                            pattern = r'\b' + re.escape(ingredient.lower()) + r'\b'
                            if not (re.search(pattern, recipe['ingredients'].lower()) or 
                                    re.search(pattern, recipe['title'].lower())):
                                all_ingredients_found = False
                                break
                        if all_ingredients_found:
                            break
                
                if len(ingredients) > 1 and not all_ingredients_found:
                    return {
                        "type": "filtered_recipes",
                        "message": f"Đây là các món ăn có chứa ít nhất một trong các nguyên liệu: {ingredient_str}{time_str}. Tôi không tìm thấy món nào chứa tất cả các nguyên liệu này cùng lúc.",
                        "recipes": clean_results,
                        "time_range": time_range,
                        "ingredients": ingredients
                    }
                else:
                    return {
                        "type": "filtered_recipes",
                        "message": f"Đây là các món ăn có nguyên liệu là {ingredient_str}{time_str}:",
                        "recipes": clean_results,
                        "time_range": time_range,
                        "ingredients": ingredients
                    }
            else:
                return {
                    "type": "not_found",
                    "message": f"Tôi không tìm thấy món ăn nào có nguyên liệu: {ingredient_str}{time_str}. Bạn có thể thử với ít nguyên liệu hơn hoặc các nguyên liệu khác.",
                    "time_range": time_range,
                    "ingredients": ingredients
                }
        elif time_range:
            conversation_manager.update_session(
                session_id, 
                state='ask_ingredients', 
                context={'time_range': time_range}
            )
            
            return {
                "type": "ask_ingredients",
                "message": f"Tôi đã ghi nhận bạn muốn món ăn trong khoảng thời gian {time_range}. Bạn muốn sử dụng nguyên liệu gì? Bạn có thể liệt kê nhiều nguyên liệu như 'thịt gà và rau cải' hoặc 'gà, tỏi, hành'.",
                "time_range": time_range
            }
        else:
            conversation_manager.update_session(session_id, state='ask_criteria')
            
            return {
                "type": "ask_criteria",
                "message": "Bạn muốn nấu món ăn trong bao nhiêu phút và sử dụng những nguyên liệu gì? Bạn có thể liệt kê nhiều nguyên liệu như 'thịt gà và rau cải' hoặc 'gà, tỏi, hành'.",
                "cooking_times": cooking_times
            }
    
    elif intent == 'provide_ingredients' and session['state'] == 'ask_ingredients':
        ingredients = extract_ingredients(message)
        
        if not ingredients:
            parts = re.split(r'[,\s]+và\s+|,', message)
            ingredients = [part.strip() for part in parts if part.strip()]
            
            if not ingredients:
                ingredients = [message.strip()]
        
        time_range = session['context'].get('time_range')
        
        results = recommender.filter_recipes_by_time_and_ingredients(time_range, ingredients)
        clean_results = clean_recipe_data(results)
        
        conversation_manager.update_session(
            session_id, 
            state='filtered', 
            context={
                'time_range': time_range, 
                'ingredients': ingredients,
                'last_ingredients': ingredients
            },
            last_recipes=clean_results
        )
        
        if clean_results:
            if len(ingredients) > 1:
                all_ingredients_message = ""
                has_all_ingredients = False
                
                for recipe in clean_results:
                    all_found = True
                    for ingredient in ingredients:
                        pattern = r'\b' + re.escape(ingredient.lower()) + r'\b'
                        if not (re.search(pattern, recipe['ingredients'].lower()) or 
                                re.search(pattern, recipe['title'].lower())):
                            all_found = False
                            break
                    if all_found:
                        has_all_ingredients = True
                        break
                
                if not has_all_ingredients:
                    all_ingredients_message = " Tôi không tìm thấy món nào chứa tất cả các nguyên liệu này cùng lúc, nên đây là các món có ít nhất một nguyên liệu trong số đó."
                
                return {
                    "type": "filtered_recipes",
                    "message": f"Đây là các món ăn phù hợp với thời gian {time_range} phút và có các nguyên liệu: {', '.join(ingredients)}.{all_ingredients_message}",
                    "recipes": clean_results,
                    "time_range": time_range,
                    "ingredients": ingredients
                }
            else:
                return {
                    "type": "filtered_recipes",
                    "message": f"Đây là các món ăn phù hợp với thời gian {time_range} phút và có nguyên liệu: {', '.join(ingredients)}:",
                    "recipes": clean_results,
                    "time_range": time_range,
                    "ingredients": ingredients
                }
        else:
            return {
                "type": "not_found",
                "message": f"Tôi không tìm thấy món ăn nào phù hợp với thời gian {time_range} phút và có nguyên liệu: {', '.join(ingredients)}. Bạn có thể thử với ít nguyên liệu hơn hoặc các nguyên liệu khác.",
                "time_range": time_range,
                "ingredients": ingredients
            }
    
    elif intent == 'search_recipe':
        ingredients = current_ingredients
        
        if ingredients:
            results = recommender.filter_recipes_by_time_and_ingredients(None, ingredients)
            clean_results = clean_recipe_data(results)
            
            conversation_manager.update_session(
                session_id, 
                state='filtered', 
                context={
                    'ingredients': ingredients,
                    'last_ingredients': ingredients
                },
                last_recipes=clean_results
            )
            
            if clean_results:
                if len(ingredients) > 1:
                    all_ingredients_message = ""
                    has_all_ingredients = False
                    
                    for recipe in clean_results:
                        all_found = True
                        for ingredient in ingredients:
                            pattern = r'\b' + re.escape(ingredient.lower()) + r'\b'
                            if not (re.search(pattern, recipe['ingredients'].lower()) or 
                                    re.search(pattern, recipe['title'].lower())):
                                all_found = False
                                break
                        if all_found:
                            has_all_ingredients = True
                            break
                    
                    if not has_all_ingredients:
                        all_ingredients_message = " Tôi không tìm thấy món nào chứa tất cả các nguyên liệu này cùng lúc, nên đây là các món có ít nhất một nguyên liệu trong số đó."
                    
                    return {
                        "type": "filtered_recipes",
                        "message": f"Đây là các món ăn có nguyên liệu là {', '.join(ingredients)}.{all_ingredients_message}",
                        "recipes": clean_results,
                        "ingredients": ingredients
                    }
                else:
                    return {
                        "type": "filtered_recipes",
                        "message": f"Đây là các món ăn có nguyên liệu là {', '.join(ingredients)}:",
                        "recipes": clean_results,
                        "ingredients": ingredients
                    }
            else:
                return {
                    "type": "not_found",
                    "message": f"Tôi không tìm thấy món ăn nào có nguyên liệu: {', '.join(ingredients)}. Bạn có thể thử với các nguyên liệu khác.",
                    "ingredients": ingredients
                }
        else:
            results = recommender.find_similar_recipe_by_text(message)
            clean_results = clean_recipe_data(results)
            
            conversation_manager.update_session(session_id, state='search_results', last_recipes=clean_results)
            
            if clean_results:
                return {
                    "type": "search_results",
                    "message": f"Đây là các món ăn phù hợp với yêu cầu của bạn:",
                    "recipes": clean_results
                }
            else:
                return {
                    "type": "not_found",
                    "message": "Tôi không tìm thấy món ăn nào phù hợp với yêu cầu của bạn. Bạn có thể thử lại với các từ khóa khác hoặc yêu cầu gợi ý món theo nguyên liệu.",
                }
    
    elif intent == 'help':
        return {
            "type": "help",
            "message": "Tôi có thể giúp bạn tìm công thức món ăn. Bạn có thể hỏi tôi về:",
            "options": [
                "Món ăn có nhiều nguyên liệu cùng lúc (Ví dụ: món có thịt gà và tỏi)",
                "Món ăn tương tự với [tên món]",
                "Món ăn có thời gian nấu dưới 15 phút",
                "Món ăn có nguyên liệu như thịt gà, rau củ, hành tỏi",
                "Danh sách các mô hình gợi ý món ăn"
            ]
        }
    
    else:
        print(f"Xử lý tổng quát cho: '{message}'")
        
        ingredients = current_ingredients
        
        if not ingredients and session['context'].get('last_ingredients'):
            ingredients = session['context']['last_ingredients']
        
        if ingredients:
            print(f"Tìm món ăn với nguyên liệu: {ingredients}")
            results = recommender.filter_recipes_by_time_and_ingredients(None, ingredients)
            
            if not results:
                print("Không tìm thấy món ăn với nguyên liệu, chuyển sang tìm kiếm theo text")
                results = recommender.find_similar_recipe_by_text(message)
        else:
            results = recommender.find_similar_recipe_by_text(message)
        
        clean_results = clean_recipe_data(results)
        print(f"Số lượng món tìm thấy: {len(clean_results)}")
        
        conversation_manager.update_session(
            session_id, 
            state='general_query', 
            context={
                'last_ingredients': ingredients or [],
            },
            last_recipes=clean_results
        )
        
        if ingredients and clean_results:
            if len(ingredients) > 1:
                all_ingredients_message = ""
                has_all_ingredients = False
                
                for recipe in clean_results:
                    all_found = True
                    for ingredient in ingredients:
                        pattern = r'\b' + re.escape(ingredient.lower()) + r'\b'
                        if not (re.search(pattern, recipe['ingredients'].lower()) or 
                                re.search(pattern, recipe['title'].lower())):
                            all_found = False
                            break
                    if all_found:
                        has_all_ingredients = True
                        break
                
                if not has_all_ingredients:
                    all_ingredients_message = " Tôi không tìm thấy món nào chứa tất cả các nguyên liệu này cùng lúc, nên đây là các món có ít nhất một nguyên liệu trong số đó."
                
                return {
                    "type": "filtered_recipes",
                    "message": f"Đây là các món ăn có nguyên liệu là {', '.join(ingredients)}.{all_ingredients_message}",
                    "recipes": clean_results,
                    "ingredients": ingredients
                }
            else:
                return {
                    "type": "filtered_recipes",
                    "message": f"Đây là các món ăn có nguyên liệu là {', '.join(ingredients)}:",
                    "recipes": clean_results,
                    "ingredients": ingredients
                }
        elif clean_results:
            return {
                "type": "search_results",
                "message": f"Đây là một số món ăn có thể phù hợp với yêu cầu của bạn:",
                "recipes": clean_results
            }
        else:
            return {
                "type": "not_found",
                "message": "Tôi không tìm thấy món ăn nào phù hợp với yêu cầu của bạn. Bạn có thể thử lại với các từ khóa khác hoặc yêu cầu gợi ý món theo nguyên liệu cụ thể.",
            }

@app.route('/')
def index():
    return render_template('chatbot.html', recipes=df['title'].tolist())

@app.route('/api/message', methods=['POST'])
def message():
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        print(f"Nhận tin nhắn từ người dùng: '{user_message}'")
        
        response = process_message(user_message, session_id)
        
        print(f"Phản hồi: type={response.get('type')}, có {len(response.get('recipes', []))} món ăn")
        
        if response.get('type') in ['similar_recipes', 'random_recipes', 'filtered_recipes', 'search_results']:
            recipes = response.get('recipes', [])
            if not recipes:
                print("Không có recipes trong phản hồi!")
            else:
                print(f"Trả về {len(recipes)} món ăn")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Lỗi xử lý tin nhắn: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "type": "error",
            "message": "Có lỗi xảy ra khi xử lý yêu cầu của bạn. Vui lòng thử lại sau.",
            "error": str(e)
        }), 500

@app.route('/api/similar_recipes', methods=['POST'])
def get_similar_recipes():
    try:
        data = request.json
        title = data.get('title', '')
        model = data.get('model', 'doc2vec')
        
        recommendations = recommender.get_similar_recipes(title, model)
        clean_recommendations = clean_recipe_data(recommendations)
        
        return jsonify({
            "success": True,
            "recipes": clean_recommendations
        })
    except Exception as e:
        print(f"Lỗi khi lấy món ăn tương tự: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Có lỗi xảy ra khi lấy món ăn tương tự",
            "error": str(e)
        }), 500

@app.route('/api/filter_recipes', methods=['POST'])
def filter_recipes_api():
    try:
        data = request.json
        time_range = data.get('time_range', '')
        ingredients = data.get('ingredients', '')
        
        results = recommender.filter_recipes_by_time_and_ingredients(time_range, ingredients)
        clean_results = clean_recipe_data(results)
        
        return jsonify({
            "success": True,
            "recipes": clean_results
        })
    except Exception as e:
        print(f"Lỗi khi lọc món ăn: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Có lỗi xảy ra khi lọc món ăn",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)