import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch

# --------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# --------------------------------------------------------
st.set_page_config(
    page_title="Reconóceme esto",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------------
# ESTILOS PERSONALIZADOS
# --------------------------------------------------------
st.markdown("""
    <style>
    body {
        background-color: white;
        color: black;
        font-family: 'Helvetica', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #f7f7f7;
        color: black;
    }
    h1, h2, h3, h4 {
        color: #111 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# CARGA DEL MODELO YOLOv5
# --------------------------------------------------------
@st.cache_resource
def load_yolov5_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        Recomendaciones:
        1. Verifica tu conexión a Internet (Streamlit Cloud la tiene por defecto)
        2. Intenta recargar la app si el modelo no carga la primera vez
        3. Si persiste, revisa la consola de logs en 'Manage App'
        """)
        return None

# --------------------------------------------------------
# ENCABEZADO Y NARRATIVA
# --------------------------------------------------------
st.title("🧠 Reconóceme esto")
st.markdown("""
### Experimento sobre cómo una red neuronal nombra el mundo  
Esta aplicación usa **YOLOv5** —una red de visión artificial— para identificar objetos visibles en imágenes.  
Cada vez que subes o capturas una foto, la IA intenta traducir la realidad en palabras.  
""")

# --------------------------------------------------------
# CARGAR MODELO
# --------------------------------------------------------
with st.spinner("Cargando modelo de visión..."):
    model = load_yolov5_model()

# --------------------------------------------------------
# FUNCIONALIDAD PRINCIPAL
# --------------------------------------------------------
if model:
    st.sidebar.title("Parámetros de detección")
    model.conf = st.sidebar.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
    model.iou = st.sidebar.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
    st.sidebar.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")

    model.agnostic = st.sidebar.checkbox('NMS class-agnostic', False)
    model.multi_label = st.sidebar.checkbox('Múltiples etiquetas por caja', False)
    model.max_det = st.sidebar.number_input('Detecciones máximas', 10, 2000, 1000, 10)

    # Subir o capturar imagen
    st.markdown("### 📷 Sube una imagen o usa la cámara")
    col1, col2 = st.columns(2)
    with col1:
        picture = st.camera_input("Capturar desde la cámara")
    with col2:
        uploaded_file = st.file_uploader("Subir una imagen", type=["jpg", "jpeg", "png"])

    img_source = None
    if picture:
        img_source = picture.getvalue()
    elif uploaded_file:
        img_source = uploaded_file.getvalue()

    if img_source:
        cv2_img = cv2.imdecode(np.frombuffer(img_source, np.uint8), cv2.IMREAD_COLOR)
        with st.spinner("Analizando la imagen..."):
            try:
                results = model(cv2_img)
            except Exception as e:
                st.error(f"Error durante la detección: {str(e)}")
                st.stop()

        # Resultados
        predictions = results.pred[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("📸 Imagen analizada")
            results.render()
            st.image(results.ims[0], channels='BGR', use_container_width=True)

        with col4:
            st.subheader("🗂️ Objetos reconocidos")

            label_names = model.names
            category_count = {}
            for category in categories:
                idx = int(category.item()) if hasattr(category, 'item') else int(category)
                category_count[idx] = category_count.get(idx, 0) + 1

            data = []
            for category, count in category_count.items():
                label = label_names[category]
                confidence = scores[categories == category].mean().item() if len(scores) > 0 else 0
                data.append({
                    "Categoría": label,
                    "Cantidad": count,
                    "Confianza promedio": f"{confidence:.2f}"
                })

            if data:
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index('Categoría')['Cantidad'])
            else:
                st.info("No se detectaron objetos. Prueba otra imagen.")
else:
    st.error("❌ No se pudo cargar el modelo. Verifica dependencias o conexión e inténtalo nuevamente.")

# --------------------------------------------------------
# PIE DE PÁGINA
# --------------------------------------------------------
st.markdown("---")
st.caption("""
**“Reconóceme esto”** — Experimento visual sobre cómo una red neuronal nombra el mundo.  
Creado por **Isabela Aristizábal** • YOLOv5 + Streamlit + PyTorch.
""")
