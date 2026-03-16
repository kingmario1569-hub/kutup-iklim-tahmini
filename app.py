import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sayfa Başlığı
st.title("❄️ Kutup İklimi ve Küresel Isınma Tahmin Paneli")
st.markdown("Verileri kutucuklara girerek beklenen sıcaklık değişimini hesaplayabilirsiniz.")

# 1. VERİLERİ YÜKLEME
@st.cache_data
def verileri_yukle():
    # Dosya isimlerinin GitHub'dakiyle aynı olduğundan emin ol
    df_buz = pd.read_excel('Sea_Ice_Index_Monthly_Data_by_Year_G02135_v4.0.xlsx')
    df_co2 = pd.read_csv('co2_annmean_mlo.txt', sep=r'\s+', comment='#', header=None, names=['Year', 'Mean'])
    df_sicaklik = pd.read_csv('sıcaklık ZonAnn.Ts+dSST (1).csv')
    return df_buz, df_co2, df_sicaklik

try:
    df_buz, df_co2, df_sicaklik = verileri_yukle()

    # Veri Temizleme ve Birleştirme
    df_buz_annual = df_buz[['Year', 'Annual']].rename(columns={'Annual': 'Ice_Extent'})
    df_sicaklik_annual = df_sicaklik[['Year', 'Glob']].rename(columns={'Glob': 'Temp_Anom'})
    
    data = pd.merge(df_co2, df_buz_annual, on='Year')
    data = pd.merge(data, df_sicaklik_annual, on='Year')

    # 2. MODEL EĞİTİMİ (Yapay Zeka)
    X = data[['Mean', 'Ice_Extent']] # CO2 ve Buz verileri (Girdi)
    y = data['Temp_Anom']           # Sıcaklık (Çıktı)
    model = LinearRegression()
    model.fit(X, y)

    # 3. YAN PANEL - SAYISAL GİRİŞ KUTUCUKLARI
    st.sidebar.header("Tahmin İçin Değerleri Girin")
    
    # Burada slider yerine rakam yazabileceğin kutucuklar (number_input) kullandık
    user_co2 = st.sidebar.number_input("CO2 Seviyesi (ppm):", value=415.0, step=0.1)
    user_ice = st.sidebar.number_input("Deniz Buzu Miktarı (milyon km²):", value=10.0, step=0.1)
    
    # Tahmin Yapma
    tahmin = model.predict([[user_co2, user_ice]])

    # 4. SONUÇLARI GÖSTERME
    st.subheader("Tahmin Sonucu")
    st.info(f"Girilen CO2 ({user_co2}) ve Buz ({user_ice}) değerlerine göre;")
    st.metric(label="Beklenen Sıcaklık Değişimi", value=f"{tahmin[0]:.3f} °C")

    # Grafik Gösterimi
    st.subheader("Geçmiş Veriler ve Trend")
    fig, ax = plt.subplots()
    ax.scatter(data['Mean'], data['Temp_Anom'], color='red', alpha=0.5)
    ax.set_xlabel("CO2 Seviyesi")
    ax.set_ylabel("Sıcaklık Değişimi")
    st.pyplot(fig)

except Exception as e:
    st.error(f"Bir hata oluştu: {e}")
