import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sayfa Yapılandırması
st.set_page_config(page_title="Kutup İklim Tahmini", layout="wide")

st.title("❄️ Kutup İklimi ve Küresel Isınma Tahmin Paneli")
st.markdown("Verileri aşağıdaki kutucuklara girerek beklenen sıcaklık değişimini anlık olarak görebilirsiniz.")

# 1. VERİLERİ YÜKLEME
@st.cache_data
def verileri_yukle():
    # Dosya isimleri (GitHub'dakiyle birebir aynı olmalı)
    df_buz = pd.read_excel('Sea_Ice_Index_Monthly_Data_by_Year_G02135_v4.0.xlsx')
    df_co2 = pd.read_csv('co2_annmean_mlo.txt', sep=r'\s+', comment='#', header=None, names=['Year', 'Mean'])
    df_sicaklik = pd.read_csv('sıcaklık ZonAnn.Ts+dSST (1).csv')
    return df_buz, df_co2, df_sicaklik

try:
    df_buz, df_co2, df_sicaklik = verileri_yukle()

    # Veri Hazırlama (Hata veren 'Year' kısmını düzeltiyoruz)
    # Excel dosyasında ilk sütun genellikle yıldır, ismini Year yapalım
    df_buz.columns = ['Year'] + list(df_buz.columns[1:])
    df_buz_annual = df_buz[['Year', 'Annual']].rename(columns={'Annual': 'Ice_Extent'})
    
    df_sicaklik_annual = df_sicaklik[['Year', 'Glob']].rename(columns={'Glob': 'Temp_Anom'})
    
    # Verileri birleştirme
    data = pd.merge(df_co2, df_buz_annual, on='Year')
    data = pd.merge(data, df_sicaklik_annual, on='Year')

    # 2. MODEL EĞİTİMİ
    X = data[['Mean', 'Ice_Extent']]
    y = data['Temp_Anom']
    model = LinearRegression()
    model.fit(X, y)

    # 3. KUTUCUKLAR (Input Boxes) - Yan tarafta
    st.sidebar.header("🎛️ Tahmin Parametreleri")
    st.sidebar.write("Değerleri kutucuklara yazın:")
    
    # Sürgü yerine rakam yazılan kutucuklar
    input_co2 = st.sidebar.number_input("Atmosferik CO2 (ppm):", min_value=300.0, max_value=600.0, value=415.0, step=0.1)
    input_ice = st.sidebar.number_input("Deniz Buzu (Milyon km²):", min_value=5.0, max_value=15.0, value=10.0, step=0.1)
    
    # Tahmin hesaplama
    tahmin_sonuc = model.predict([[input_co2, input_ice]])[0]

    # 4. GÖRSELLEŞTİRME VE SONUÇ
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Tahmin Sonucu")
        st.metric(label="Beklenen Sıcaklık Artışı", value=f"{tahmin_sonuc:.3f} °C")
        st.write("---")
        st.write(f"Girilen CO2: **{input_co2}**")
        st.write(f"Girilen Buz: **{input_ice}**")

    with col2:
        st.subheader("Geçmişten Günümüze Durum")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(data['Year'], data['Temp_Anom'], color='red', marker='o', label='Gerçek Veri')
        ax.set_xlabel("Yıl")
        ax.set_ylabel("Sıcaklık Artışı (°C)")
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)

except Exception as e:
    st.error(f"Sistemde bir hata var: {e}")
    st.info("Lütfen GitHub'daki dosya isimlerinin ve sütun başlıklarının doğruluğunu kontrol edin.")
