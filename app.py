import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="10 Yıllık İklim Tahmini", layout="wide")

# Görsel Stil: Kutucukları ve yazılarını büyütmek için CSS
st.markdown("""
    <style>
    .stNumberInput label {font-size: 20px !important; font-weight: bold; color: #2e86c1;}
    .stNumberInput input {font-size: 25px !important; height: 60px !important;}
    .main-title {font-size: 40px !important; text-align: center; color: #1b4f72; margin-bottom: 30px;}
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">❄️ 10 Yıllık Küresel Isınma Projeksiyonu</p>', unsafe_allow_html=True)

@st.cache_data
def verileri_yukle():
    df_co2 = pd.read_csv('co2_annmean_mlo.txt', sep=r'\s+', comment='#', header=None)
    df_co2.columns = ['Year', 'Mean', 'Unused']
    df_buz = pd.read_excel('Sea_Ice_Index_Monthly_Data_by_Year_G02135_v4.0.xlsx')
    df_sicaklik = pd.read_csv('sıcaklık ZonAnn.Ts+dSST (1).csv')
    return df_buz, df_co2, df_sicaklik

try:
    df_buz, df_co2, df_sicaklik = verileri_yukle()
    
    # Veri Hazırlama
    df_co2['Year'] = df_co2['Year'].astype(int)
    df_buz['Year'] = pd.to_numeric(df_buz.iloc[:, 0], errors='coerce').fillna(0).astype(int)
    df_sicaklik['Year'] = df_sicaklik['Year'].astype(int)

    # Birleştirme
    data = pd.merge(df_co2[['Year', 'Mean']], df_sicaklik[['Year', 'Glob']], on='Year')
    data = pd.merge(data, df_buz[['Year', 'Annual']], on='Year')
    data.columns = ['Year', 'CO2', 'Temp', 'Ice']

    # 10 Yıl Sonrası Modeli
    data['Target_Temp'] = data['Temp'].shift(-10)
    train_data = data.dropna()
    X = train_data[['CO2', 'Temp', 'Ice']]
    y = train_data['Target_Temp']
    model = LinearRegression().fit(X, y)

    # --- ANA SAYFA GİRİŞ ALANI (ORTADA VE BÜYÜK) ---
    st.write("### 📡 Güncel Verileri Girin")
    st.info("Aşağıdaki kutucuklara güncel değerleri yazarak 10 yıl sonrasını tahmin edin.")
    
    # Giriş kutularını yan yana 3 sütuna bölüyoruz
    c1, c2, c3 = st.columns(3)
    
    with c1:
        guncel_co2 = st.number_input("CO2 Seviyesi (ppm)", 300.0, 600.0, 420.0)
    with c2:
        guncel_sicaklik = st.number_input("Sıcaklık Artışı (°C)", -1.0, 3.0, 1.2)
    with c3:
        guncel_buz = st.number_input("Buz Kütlesi (M km²)", 5.0, 15.0, 10.2)

    st.write("---") # Ayırıcı çizgi

    # Tahmin Hesaplama
    tahmin_10_yil = model.predict([[guncel_co2, guncel_sicaklik, guncel_buz]])[0]

    # --- SONUÇ PANELİ ---
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.subheader("🔮 10 Yıl Sonraki Tahmin")
        # Metric değerini büyütüyoruz
        st.metric(
            label="Tahmini Sıcaklık Artışı (2036)", 
            value=f"{tahmin_10_yil:.3f} °C", 
            delta=f"{tahmin_10_yil - guncel_sicaklik:.3f} °C Değişim"
        )
        
    with res_col2:
        st.subheader("📊 Analiz")
        if tahmin_10_yil > 1.5:
            st.error(f"Kritik Sınır Aşılıyor! Tahmin: {tahmin_10_yil:.2f}°C")
        else:
            st.success("Sıcaklık artışı kontrol edilebilir seviyede.")

    # Grafik
    st.write("---")
    st.subheader("📈 Tarihsel Veri Akışı")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data['Year'], data['Temp'], color='#e74c3c', label="Sıcaklık")
    ax.fill_between(data['Year'], data['Temp'], color='#e74c3c', alpha=0.1)
    ax.set_title("Geçmişten Günümüze Sıcaklık Değişimi")
    st.pyplot(fig)

except Exception as e:
    st.error(f"Hata: {e}")
