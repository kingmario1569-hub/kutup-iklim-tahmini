import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Kutup İklim Tahmini", layout="wide")
st.title("❄️ Kutup İklimi ve Küresel Isınma Tahmin Paneli")

@st.cache_data
def verileri_yukle():
    # CO2 dosyasını doğru sütunlarla okuyalım (İlk sütun yıl, ikinci sütun değer)
    df_co2 = pd.read_csv('co2_annmean_mlo.txt', sep=r'\s+', comment='#', header=None)
    df_co2.columns = ['Year', 'Mean', 'Unused'] # İlk iki sütunu aldık
    
    df_buz = pd.read_excel('Sea_Ice_Index_Monthly_Data_by_Year_G02135_v4.0.xlsx')
    df_sicaklik = pd.read_csv('sıcaklık ZonAnn.Ts+dSST (1).csv')
    return df_buz, df_co2, df_sicaklik

try:
    df_buz, df_co2, df_sicaklik = verileri_yukle()

    # Veri Tiplerini Zorla Düzenle
    df_co2['Year'] = pd.to_numeric(df_co2['Year'], errors='coerce').astype(float).fillna(0).astype(int)
    df_buz['Year'] = pd.to_numeric(df_buz.iloc[:, 0], errors='coerce').fillna(0).astype(int)
    df_sicaklik['Year'] = pd.to_numeric(df_sicaklik['Year'], errors='coerce').fillna(0).astype(int)

    # Birleştirme (Inner Join)
    # Önce CO2 ve Sıcaklık
    data = pd.merge(df_co2[['Year', 'Mean']], df_sicaklik[['Year', 'Glob']], on='Year', how='inner')
    # Sonra Buz (Annual sütunu)
    data = pd.merge(data, df_buz[['Year', 'Annual']], on='Year', how='inner')
    
    data = data.rename(columns={'Mean': 'CO2', 'Annual': 'Ice_Extent', 'Glob': 'Temp_Anom'})

    if data.empty:
        st.error("⚠️ Veriler eşleşemedi. Lütfen dosyaları kontrol edin.")
        # Debug bilgisi
        st.write("CO2 Dosyasından okunan ilk 3 yıl:", df_co2['Year'].head(3).values)
        st.write("Buz Dosyasından okunan ilk 3 yıl:", df_buz['Year'].head(3).values)
    else:
        # MODEL EĞİTİMİ
        X = data[['CO2', 'Ice_Extent']]
        y = data['Temp_Anom']
        model = LinearRegression().fit(X, y)

        # ARAYÜZ
        st.sidebar.header("🎛️ Tahmin Parametreleri")
        input_co2 = st.sidebar.number_input("Atmosferik CO2 (ppm):", 300.0, 550.0, 415.0)
        input_ice = st.sidebar.number_input("Deniz Buzu (Milyon km²):", 5.0, 15.0, 10.0)
        
        tahmin = model.predict([[input_co2, input_ice]])[0]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("🤖 Sonuç")
            st.metric("Tahmin Edilen Sıcaklık Artışı", f"{tahmin:.3f} °C")
        
        with col2:
            st.subheader("📈 Sıcaklık Trendi")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(data['Year'], data['Temp_Anom'], 'r-o', label="Gözlemlenen")
            ax.set_xlabel("Yıl"); ax.set_ylabel("Artış (°C)")
            st.pyplot(fig)

except Exception as e:
    st.error(f"Hata detayı: {e}")
