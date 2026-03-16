import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. SAYFA AYARLARI VE BAŞLIK
st.set_page_config(page_title="Kutup Analizi", layout="wide")
st.title("🌍 Kutup Bölgeleri ve Ekstrem Sıcaklık Tahmin Uygulaması")
st.markdown("Bu uygulama, NASA ve NSIDC verilerini kullanarak makine öğrenmesi ile gelecek tahminleri yapar.")

# 2. VERİLERİ YÜKLE VE TEMİZLE (Colab'daki mevcut dosyalarını okur)
@st.cache_data
def verileri_hazirla():
    df_buz = pd.read_excel('Sea_Ice_Index_Monthly_Data_by_Year_G02135_v4.0.xlsx').rename(columns={'Unnamed: 0': 'Year'})
    df_co2 = pd.read_csv('co2_annmean_mlo.txt', sep=r'\s+', comment='#', header=None, names=['Year', 'Mean', 'Uncertainty'])
    df_sicaklik = pd.read_csv('sıcaklık ZonAnn.Ts+dSST (1).csv')
    
    # Birleştirme (Senin dediğin gibi yılları eşitliyoruz)
    ara_tablo = pd.merge(df_buz[['Year', 'Annual']], df_co2[['Year', 'Mean']], on='Year')
    final_tablo = pd.merge(ara_tablo, df_sicaklik[['Year', 'Glob']], on='Year')
    return final_tablo

data = verileri_hazirla()

# 3. MAKİNE ÖĞRENMESİ MODELİNİ EĞİT
X = data[['Mean', 'Annual']]
y = data['Glob']
model = LinearRegression()
model.fit(X, y)

# 4. ARAYÜZ TASARIMI (Sol Panel)
st.sidebar.header("Gelecek Senaryosu Oluştur")
st.sidebar.write("Aşağıdaki değerleri değiştirerek sıcaklık tahminini gör:")
user_co2 = st.sidebar.slider("Atmosferik CO2 (ppm)", 300, 500, int(data['Mean'].max()))
user_buz = st.sidebar.slider("Buz Miktarı (Milyon km²)", 5.0, 15.0, float(data['Annual'].mean()))

# 5. TAHMİN VE GÖSTERGE
tahmin_sonucu = model.predict([[user_co2, user_buz]])

col1, col2 = st.columns(2)
with col1:
    st.subheader("Tahmin Sonucu")
    st.metric(label="Beklenen Sıcaklık Artışı", value=f"{tahmin_sonucu[0]:.2f} °C")
    st.info("Bu tahmin, girdiğiniz CO2 ve Buz miktarına göre makine öğrenmesi modelimiz tarafından hesaplanmıştır.")

with col2:
    st.subheader("Veri Özeti")
    st.write(data.tail(5)) # Son 5 yılı gösterir

# 6. GRAFİK
st.subheader("Geçmişten Günümüze Sıcaklık Değişimi")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data['Year'], data['Glob'], color='red', marker='o')
ax.set_xlabel("Yıl")
ax.set_ylabel("Sıcaklık Artışı (°C)")
st.pyplot(fig)
