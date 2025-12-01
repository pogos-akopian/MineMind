import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- 1. CONFIGURATION & TITLE ---
st.set_page_config(page_title="MineMind MVP", page_icon="⛏️")

st.title("⛏️ MineMind: Intelligent Asset Manager")
st.markdown("""
**Система предиктивного обслуживания карьерной техники.**
Используйте панель слева для симуляции показателей датчиков.
""")

# --- 2. DATA LOADING & TRAINING (MOCKING REALITY) ---
@st.cache_data
def load_and_train():
    # Загружаем "чистые" данные
    try:
        df = pd.read_csv('ai4i2020.csv')
    except FileNotFoundError:
        st.error("Файл ai4i2020.csv не найден. Пожалуйста, загрузите его в репозиторий.")
        return None, None

    # Переименовываем колонки под легенду Mining
    # Air Temp -> Ambient Temp (Шахта)
    # Process Temp -> Engine Temp (Двигатель)
    # Torque -> Load (Нагрузка)
    df.rename(columns={
        'Air temperature [K]': 'Ambient Temp',
        'Process temperature [K]': 'Engine Temp',
        'Rotational speed [rpm]': 'RPM',
        'Torque [Nm]': 'Load',
        'Tool wear [min]': 'Drill Bit Wear',
        'Machine failure': 'Failure'
    }, inplace=True)

    # Простая предобработка
    le = LabelEncoder()
    df['Type'] = le.fit_transform(df['Type']) # L/M/H -> 0/1/2

    # Выбираем фичи для модели
    features = ['Type', 'Ambient Temp', 'Engine Temp', 'RPM', 'Load', 'Drill Bit Wear']
    X = df[features]
    y = df['Failure']

    # Обучаем простую модель (Random Forest)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    return model, le

model, le = load_and_train()

if model is not None:
    # --- 3. SIDEBAR (CONTROLS) ---
    st.sidebar.header("⚙️ Панель телеметрии")
    
    # Симуляция "Твердости породы"
    rock_type = st.sidebar.selectbox("Тип породы (Rock Hardness)", ['Low (Sandstone)', 'Medium (Limestone)', 'High (Granite)'])
    type_map = {'Low (Sandstone)': 'L', 'Medium (Limestone)': 'M', 'High (Granite)': 'H'}
    # Кодируем выбор обратно в цифры для модели (L=1, M=2, H=0 - примерная логика LabelEncoder)
    # Для простоты MVP просто передадим среднее значение, если маппинг сложный, 
    # но здесь для демонстрации оставим слайдеры.
    
    # Слайдеры датчиков
    ambient = st.sidebar.slider("Температура в забое (K)", 290, 310, 300)
    engine_temp = st.sidebar.slider("Температура двигателя (K)", 300, 340, 310)
    rpm = st.sidebar.slider("Обороты бура (RPM)", 1100, 2900, 1500)
    load = st.sidebar.slider("Нагрузка на привод (Nm)", 0, 80, 40)
    wear = st.sidebar.slider("Износ коронки (min)", 0, 300, 0)

    # --- 4. PREDICTION LOGIC ---
    # Превращаем ввод пользователя в данные для модели
    # Примечание: LabelEncoder в MVP может кодировать L/M/H динамически, для упрощения возьмем 1 (Medium)
    type_val = 1 
    if rock_type == 'Low (Sandstone)': type_val = 1 # В датасете L - самый частый
    elif rock_type == 'Medium (Limestone)': type_val = 2
    else: type_val = 0
    
    input_data = pd.DataFrame({
        'Type': [type_val],
        'Ambient Temp': [ambient],
        'Engine Temp': [engine_temp],
        'RPM': [rpm],
        'Load': [load],
        'Drill Bit Wear': [wear]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # --- 5. MAIN DISPLAY ---
    
    st.subheader("📊 Статус оборудования")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Текущий износ", f"{wear} min")
    col2.metric("Нагрузка", f"{load} Nm")
    col3.metric("Температура", f"{engine_temp} K")

    st.divider()

    # Визуализация прогноза
    if prediction == 1 or probability > 0.5:
        st.error(f"⚠️ ВНИМАНИЕ: Высокий риск аварии! (Вероятность: {probability:.1%})")
        
        # --- BUSINESS LOGIC (Economic Optimizer) ---
        st.subheader("💰 Economic Optimizer")
        st.write("Система рассчитала оптимальное действие:")
        
        cost_maintenance = 500  # $ Стоимость планового ремонта
        cost_failure = 20000    # $ Стоимость аварии
        
        expected_loss = cost_failure * probability
        
        col_A, col_B = st.columns(2)
        
        with col_A:
            st.info(f"📉 Стоимость превентивного ремонта: **${cost_maintenance}**")
            st.button("🛠 Заказать ремонт сейчас")
            
        with col_B:
            st.warning(f"🔥 Ожидаемые потери при отказе: **${int(expected_loss)}**")
            
        if expected_loss > cost_maintenance:
            st.success(f"💡 РЕКОМЕНДАЦИЯ: **Остановить и чинить**. Вы сэкономите **${int(expected_loss - cost_maintenance)}**")
        else:
            st.info("💡 РЕКОМЕНДАЦИЯ: Риск допустим, можно завершить смену.")
            
    else:
        st.success(f"✅ Система в норме. Вероятность сбоя: {probability:.1%}")
        st.write("Продолжайте работу в штатном режиме.")
