import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sayfa Yapılandırması
st.set_page_config(page_title="Kutup İklim Tahmini", layout="wide")

st.title("❄️ Kutup İklimi ve Küresel Isınma Tahmin Paneli")
st.markdown("Değerleri sol taraftaki kutucuklara girerek beklenen sıcaklık değişimini hesaplayabilirsiniz.")

# 1. VERİLERİ YÜKLEME
@st.cache_data
def verileri_yukle():
    # Dosya isimleri
    df_buz = pd.read_excel('Sea_Ice_Index_Monthly_Data_by_Year_G02135_v4.0.xlsx')
    df_co2 = pd.read_csv('co2_annmean_mlo.txt', sep=r'\s+', comment='#', header=None, names=['Year', 'Mean'])
    df_sicaklik = pd.read_csv('sıcaklık ZonAnn.Ts+dSST (1).csv')
    return df_buz, df_co2, df_sicaklik

try:
    df_buz, df_co2, df_sicaklik = verileri_yukle()

    # VERİ DÜZENLEME - GARANTİ YÖNTEM
    # Buz verisi: İlk sütun yıl, 'Annual' sütunu buz miktarı
    df_buz_clean = pd.DataFrame({
        'Year': df_buz.iloc[:, 0].astype(int),
        'Ice_Extent': df_buz['Annual'].astype(float)
    })
    
    # Sıcaklık verisi: 'Year' ve 'Glob' sütunları
    df_sicaklik_clean = pd.DataFrame({
        'Year': df_sicaklik['Year'].astype(int),
        'Temp_Anom': df_sicaklik['Glob'].astype(float)
    })

    # CO2 verisi zaten düzgün, sadece yılı tam sayı yapalım
    df_co2['Year'] = df_co2['Year'].astype(int)

    # TÜM VERİLERİ BİRLEŞTİRME (Ortak yıllara göre)
    data = pd.merge(df_co2, df_buz_clean, on='Year')
    data = pd.merge(data, df_sicaklik_clean, on='Year')

    if data.empty:
        st.error("Veriler birleştirilemedi. Dosya içeriklerindeki yıllar eşleşmiyor.")
    else:
        # 2. MODEL EĞİTİMİ
        X = data[['Mean', 'Ice_Extent']]
        y = data['Temp_Anom']
        model = LinearRegression()
        model.fit(X, y)

        # 3. KUTUCUKLAR (Input Boxes) - Yan tarafta
        st.sidebar.header("🎛️ Tahmin Parametreleri")
        
        input_co2 = st.sidebar.number_input("Atmosferik CO2 (ppm):", min_value=300.0, max_value=500.0, value=415.0, step=0.1)
        input_ice = st.sidebar.number_input("Deniz Buzu (Milyon km²):", min_value=5.0, max_value=15.0, value=10.0, step=0.1)
        
        # Tahmin hesaplama
        tahmin_sonuc = model.predict([[input_co2, input_ice]])[0]

        # 4. GÖRSELLEŞTİRME VE SONUÇ
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("🤖 Tahmin Sonucu")
            st.metric(label="Tahmin Edilen Sıcaklık Artışı", value=f"{tahmin_sonuc:.3f} °C")
            st.info(f"CO2: {input_co2} ppm\n\nBuz: {input_ice} M km²")

        with col2:
            st.subheader("📈 Tarihsel Sıcaklık Değişimi")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data['Year'], data['Temp_Anom'], color='#e74c3c', linewidth=2, marker='o', markersize=4, label='Sıcaklık Artışı')
            ax.set_xlabel("Yıl")
            ax.set_ylabel("Sıcaklık Değişimi (°C)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

except Exception as e:
    st.error(f"Bir hata oluştu: {e}")
