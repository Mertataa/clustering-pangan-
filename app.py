
import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
import joblib
import pickle
import os

st.set_page_config(page_title="Peta Clustering Harga Komoditas", layout="wide")
st.title("🗺️ Peta Clustering Harga Komoditas Indonesia")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Pengaturan")
komoditas_list = ['Cabai Merah', 'Daging Ayam', 'Telur Ayam']
komoditas = st.sidebar.selectbox("Pilih Komoditas", komoditas_list)

# Load data
@st.cache_data
def load_data(komoditas):
    try:
        with open(f'models/provinsi_{komoditas.replace(" ", "_")}.pkl', 'rb') as f:
            provinsi = pickle.load(f)
        kmeans = joblib.load(f'models/kmeans_{komoditas.replace(" ", "_")}.pkl')
        return provinsi, kmeans
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

provinsi, kmeans = load_data(komoditas)

if provinsi is not None and kmeans is not None:
    labels = kmeans.labels_
    k = len(set(labels))
    
    provinsi_coords = {
        'Aceh': [4.5, 97.0], 'Sumatera Utara': [2.0, 99.0],
        'Sumatera Barat': [-1.0, 100.5], 'Riau': [0.5, 102.0],
        'Jambi': [-1.5, 102.5], 'Bengkulu': [-3.5, 102.0],
        'Lampung': [-5.0, 105.0], 'Bangka Belitung': [-2.5, 106.0],
        'Kepulauan Riau': [0.5, 104.5], 'DKI Jakarta': [-6.2, 106.8],
        'Jawa Barat': [-6.9, 107.6], 'Jawa Tengah': [-7.5, 110.0],
        'DI Yogyakarta': [-7.8, 110.4], 'Jawa Timur': [-7.5, 112.0],
        'Banten': [-6.1, 106.0], 'Bali': [-8.3, 115.2],
        'Nusa Tenggara Barat': [-8.5, 116.0], 'Nusa Tenggara Timur': [-9.0, 120.0],
        'Kalimantan Barat': [0.0, 110.0], 'Kalimantan Tengah': [-2.0, 113.0],
        'Kalimantan Selatan': [-3.0, 115.0], 'Kalimantan Timur': [0.5, 116.0],
        'Kalimantan Utara': [2.5, 116.0], 'Sulawesi Utara': [1.0, 124.0],
        'Sulawesi Tengah': [-1.0, 121.0], 'Sulawesi Selatan': [-4.0, 120.0],
        'Sulawesi Tenggara': [-4.0, 122.0], 'Gorontalo': [0.5, 123.0],
        'Sulawesi Barat': [-3.0, 119.0], 'Maluku': [-3.0, 129.0],
        'Maluku Utara': [0.5, 127.0], 'Papua': [-4.0, 138.0],
        'Papua Barat': [-1.0, 133.0],
    }
    
    provinsi_cluster = dict(zip(provinsi, labels))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    st.subheader(f"📍 Peta Clustering - {komoditas}")
    
    m = folium.Map(location=[-2.5, 118.0], zoom_start=5, tiles='OpenStreetMap')
    
    for prov, cluster_id in provinsi_cluster.items():
        if prov in provinsi_coords:
            lat, lon = provinsi_coords[prov]
            color = colors[cluster_id % len(colors)]
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=10,
                popup=f"<b>{prov}</b><br>Cluster: {cluster_id}",
                tooltip=prov,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.8,
                weight=2
            ).add_to(m)
    
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; 
                background: white; border:2px solid grey; z-index:9999;
                padding: 10px; border-radius: 10px; box-shadow: 0 0 15px rgba(0,0,0,0.3);">
        <b style="font-size: 16px;">📊 Cluster - {komoditas}</b><br><hr>
    '''
    for i in range(k):
        count = sum(1 for c in provinsi_cluster.values() if c == i)
        color = colors[i % len(colors)]
        legend_html += f'''
        <div style="display:flex; align-items:center; margin:5px 0;">
            <div style="width:20px; height:20px; background:{color}; 
                        border:1px solid black; border-radius:50%; margin-right:10px;"></div>
            <span>Cluster {i} ({count} provinsi)</span>
        </div>
        '''
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    folium_static(m, width=1000, height=550)
    
    st.markdown("---")
    st.subheader("📋 Daftar Provinsi per Cluster")
    df = pd.DataFrame({'Provinsi': provinsi, 'Cluster': labels})
    df = df.sort_values('Cluster')
    st.dataframe(df, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📊 Statistik Cluster")
    cols = st.columns(3)
    for i in range(k):
        count = sum(1 for c in provinsi_cluster.values() if c == i)
        persentase = (count / len(provinsi)) * 100
        with cols[i % 3]:
            st.metric(f"Cluster {i}", f"{count} provinsi", f"{persentase:.1f}%")
else:
    st.warning("Model belum tersedia. Pastikan folder models/ berisi file model.")
