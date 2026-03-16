import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="10 Yıllık İklim Tahmini", layout="wide")

# Görsel stil ayarları (Kutucukları biraz daha belirgin yapar)
st.markdown("""
    <style>
    .stNumberInput {border: 2px solid #3498db; border-radius: 5px;}
    </style>
    """, unsafe_allow_html=True)

st.title("❄️ 10 Yıllık Küresel Isınma Projeksiyonu")

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

    # --- 10 YIL SONRASI İÇİN MODELLEME ---
    # Modelimizi "Bugünkü veriler -> 10 yıl sonraki sıcaklık" şeklinde eğitiyoruz
    data['Target_Temp'] = data['Temp'].shift(-10) # 10 yıl sonrasının sıcaklığını hedef yap
    train_data = data.dropna() # Boş satırları (son 10 yılı) temizle

    X = train_data[['CO2', 'Temp', 'Ice']]
    y = train_data['Target_Temp']
    model = LinearRegression().fit(X, y)

    # --- YAN PANEL: 3 ANA GİRDİ ---
    st.sidebar.header("📡 Güncel Verileri Girin")
    st.sidebar.info("Girdiğiniz değerlere dayanarak 10 yıl sonrası tahmin edilir.")
    
    # İstediğin 3 Girdi Kutucuğu
    guncel_co2 = st.sidebar.number_input("Güncel CO2 Seviyesi (ppm):", 300.0, 600.0, 420.0, help="Şu anki atmosferik CO2")
    guncel_sicaklik = st.sidebar.number_input("Güncel Sıcaklık Artışı (°C):", -1.0, 3.0, 1.2, help="Şu anki küresel sıcaklık farkı")
    guncel_buz = st.sidebar.number_input("Güncel Buz Miktarı (M km²):", 5.0, 15.0, 10.2, help="Şu anki deniz buzu kütlesi")

    # Tahmin Hesaplama
    tahmin_10_yil = model.predict([[guncel_co2, guncel_sicaklik, guncel_buz]])[0]

    # --- SONUÇ EKRANI ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔮 10 Yıl Sonraki Durum")
        st.write("Yapay zeka modelimiz, girdiğiniz parametrelerin 10 yıl sonraki etkisini hesapladı:")
        st.metric(label="Tahmini Sıcaklık Artışı (2036 Tahmini)", value=f"{tahmin_10_yil:.3f} °C", delta=f"{tahmin_10_yil - guncel_sicaklik:.3f} °C Değişim")
        
        if tahmin_10_yil > 1.5:
            st.warning("⚠️ Tahmin, 1.5°C kritik eşiğinin üzerinde!")
        else:
            st.success("✅ Tahmin, kritik sınırların altında görünüyor.")

    with col2:
        st.subheader("📊 Tarihsel Veri Seti")
        st.dataframe(data.tail(10), use_container_width=True)

    # Grafik
    st.subheader("📈 Zaman İçindeki Değişim")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data['Year'], data['Temp'], label="Geçmiş Sıcaklık", color='red')
    ax.set_xlabel("Yıl")
    ax.set_ylabel("Sıcaklık (°C)")
    st.pyplot(fig)

except Exception as e:
    st.error(f"Bir hata oluştu: {e}")
