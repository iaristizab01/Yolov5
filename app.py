import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os

# Configuración de página Streamlit
st.set_page_config(
    page_title="Detección de Objetos en Imágenes",
    page_icon="🔍",
    layout="wide"
)

# --------------------------------------------------------
# FUNCIÓN PARA CARGAR EL MODELO YOLOv5
# --------------------------------------------------------
@st.cache_resource
def load_yolov5_model():
    try:
        # Cargar el modelo YOLOv5 directamente desde Ultralytics (sin necesidad de instalar yolov5)
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        Recomendaciones:
        1. Verifica tu conexión a Internet (Streamlit Cloud tiene conexión por defecto)
        2. Intenta recargar la app si el modelo no carga la primera vez
        3. Si persiste, revisa la consola de logs en 'Manage App'
        """)
        return None

# --------------------------------------------------------
# INTERFAZ PRINCIPAL
# --------------------------------------------------------
st.title("🔍 Detección de Objetos en Imágenes")
st.markdown("""
Esta aplicación utiliza **YOLOv5** para detectar objetos en imágenes capturadas con tu cámara.
Ajusta los parámetros en la barra lateral para personalizar la detección.
""")

# Cargar el modelo YOLOv5
with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

# --------------------------------------------------------
# SI EL MODELO SE CARGA CORRECTAMENTE
# --------------------------------------------------------
if model:
    st.sidebar.title("Parámetros")

    # Configuración de confianza e IoU
    model.conf = st.sidebar.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
    model.iou = st.sidebar.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
    st.sidebar.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")

    # Otras opciones
    try:
        model.agnostic = st.sidebar.checkbox('NMS class-agnostic', False)
        model.multi_label = st.sidebar.checkbox('Múltiples etiquetas por caja', False)
        model.max_det = st.sidebar.number_input('Detecciones máximas', 10, 2000, 1000, 10)
    except:
        st.sidebar.warning("Algunas opciones avanzadas no están disponibles con esta configuración")

    # --------------------------------------------------------
    # CAPTURAR IMAGEN DE LA CÁMARA
    # --------------------------------------------------------
    picture = st.camera_input("📸 Captura una imagen para analizar")

    if picture:
        # Procesar imagen capturada
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Detección con YOLOv5
        with st.spinner("Detectando objetos..."):
            try:
                results = model(cv2_img)
            except Exception as e:
                st.error(f"Error durante la detección: {str(e)}")
                st.stop()

        # --------------------------------------------------------
        # MOSTRAR RESULTADOS
        # --------------------------------------------------------
        try:
            predictions = results.pred[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4]
            categories = predictions[:, 5]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📷 Imagen con detecciones")
                results.render()
                # Mostrar la imagen con detecciones
                st.image(results.ims[0], channels='BGR', use_container_width=True)

            with col2:
                st.subheader("📋 Objetos detectados")

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
                    st.info("No se detectaron objetos con los parámetros actuales.")
                    st.caption("Prueba a reducir el umbral de confianza.")
        except Exception as e:
            st.error(f"Error al procesar los resultados: {str(e)}")
            st.stop()
else:
    st.error("❌ No se pudo cargar el modelo. Verifica dependencias o conexión e inténtalo nuevamente.")

# --------------------------------------------------------
# PIE DE PÁGINA
# --------------------------------------------------------
st.markdown("---")
st.caption("""
**Desarrollado por Isabela Aristizábal**  
App de detección de objetos con **YOLOv5 + Streamlit + PyTorch**.
""")
