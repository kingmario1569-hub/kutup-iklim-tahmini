import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sayfa Yapılandırması
st.set_page_config(page_title="Kutup İklim Tahmini", layout="wide")

st.title("❄️ Kutup İklimi ve Küresel Isınma Tahmin Paneli")

@st.cache_data
def verileri_yukle():
    df_buz = pd.read_excel('Sea_Ice_Index_Monthly_Data_by_Year_G02135_v4.0.xlsx')
    df_co2 = pd.read_csv('co2_annmean_mlo.txt', sep=r'\s+', comment='#', header=None, names=['Year', 'Mean'])
    df_sicaklik = pd.read_csv('sıcaklık ZonAnn.Ts+dSST (1).csv')
    return df_buz, df_co2, df_sicaklik

try:
    df_buz, df_co2, df_sicaklik = verileri_yukle()

    # 1. VERİ TEMİZLEME (Yılları Sayıya Çevirme)
    df_buz['Year'] = pd.to_numeric(df_buz.iloc[:, 0], errors='coerce')
    df_co2['Year'] = pd.to_numeric(df_co2['Year'], errors='coerce')
    df_sicaklik['Year'] = pd.to_numeric(df_sicaklik['Year'], errors='coerce')

    # Boş yılları temizle
    df_buz = df_buz.dropna(subset=['Year'])
    df_co2 = df_co2.dropna(subset=['Year'])
    df_sicaklik = df_sicaklik.dropna(subset=['Year'])

    # 2. BİRLEŞTİRME (Explicit Inner Join)
    # Önce CO2 ve Sıcaklık
    data = pd.merge(df_co2, df_sicaklik[['Year', 'Glob']], on='Year', how='inner')
    # Sonra Buz verisi (Buz verisindeki 'Annual' sütununu alıyoruz)
    data = pd.merge(data, df_buz[['Year', 'Annual']], on='Year', how='inner')
    data = data.rename(columns={'Annual': 'Ice_Extent', 'Glob': 'Temp_Anom'})

    if data.empty:
        st.error("⚠️ Veriler birleştirilemedi! Yıllar eşleşmiyor.")
        st.write("CO2 Yılları:", df_co2['Year'].min(), "-", df_co2['Year'].max())
        st.write("Buz Yılları:", df_buz['Year'].min(), "-", df_buz['Year'].max())
    else:
        # 3. MODEL EĞİTİMİ
        X = data[['Mean', 'Ice_Extent']]
        y = data['Temp_Anom']
        model = LinearRegression()
        model.fit(X, y)

        # 4. KUTUCUKLAR
        st.sidebar.header("🎛️ Tahmin Parametreleri")
        input_co2 = st.sidebar.number_input("Atmosferik CO2 (ppm):", 300.0, 500.0, 415.0)
        input_ice = st.sidebar.number_input("Deniz Buzu (Milyon km²):", 5.0, 15.0, 10.0)
        
        tahmin = model.predict([[input_co2, input_ice]])[0]

        # 5. EKRANA YAZDIRMA
        col1, col2 = st.columns(2)
        col1.metric("Tahmin Edilen Sıcaklık Artışı", f"{tahmin:.3f} °C")
        
        fig, ax = plt.subplots()
        ax.scatter(data['Year'], data['Temp_Anom'], color='red')
        ax.set_title("Yıllara Göre Sıcaklık Değişimi")
        st.pyplot(fig)

except Exception as e:
    st.error(f"Hata: {e}")
