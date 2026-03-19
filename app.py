# Run from project root:  streamlit run app.py
import streamlit as st, numpy as np, cv2, tensorflow as tf
from pathlib import Path
from PIL import Image

ROOT        = Path(__file__).resolve().parent
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_PATH  = ROOT / "outputs" / "best_final_model_v2.h5"

# Read IMG_SIZE from saved model input shape
@st.cache_resource
def load_model():
    m = tf.keras.models.load_model(str(MODEL_PATH))
    return m

CLASS_INFO = {
    "glioma":     ("🔴", "Glioma",     "Arises from glial cells. Most common primary brain tumour."),
    "meningioma": ("🟠", "Meningioma", "Usually benign; grows on brain membranes."),
    "notumor":    ("🟢", "No Tumour",  "No tumour detected in the MRI scan."),
    "pituitary":  ("🔵", "Pituitary",  "Located at the base of the brain; often benign adenoma."),
}

st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")
st.title("🧠 Brain Tumor MRI Classifier")
st.caption("Upload an MRI image — glioma · meningioma · pituitary · no tumour")

try:
    model = load_model()
    IMG_SIZE = tuple(model.input_shape[1:3])
    st.success(f"Model loaded  |  input: {IMG_SIZE}", icon="✅")
except Exception as e:
    st.error(f"Cannot load model: {e}"); st.stop()

uploaded = st.file_uploader("Upload MRI (JPG / PNG)", type=["jpg","jpeg","png"])
if uploaded:
    pil   = Image.open(uploaded).convert("RGB")
    arr   = np.array(pil.resize(IMG_SIZE)) / 255.0
    batch = np.expand_dims(arr, 0)
    with st.spinner("Analysing …"):
        preds   = model.predict(batch, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        top_cls = CLASS_NAMES[top_idx]
        conf    = float(preds[top_idx]) * 100
    emoji, label, desc = CLASS_INFO[top_cls]
    c1, c2 = st.columns(2)
    with c1:
        st.image(pil, caption="Uploaded MRI", width="content")
    with c2:
        st.subheader(f"{emoji}  {label}")
        st.metric("Confidence", f"{conf:.1f}%")
        st.info(desc)
        st.divider()
        for cls, p in zip(CLASS_NAMES, preds):
            st.progress(float(p), text=f"{CLASS_INFO[cls][0]} {cls}: {p*100:.1f}%")
    st.subheader("🔥 Grad-CAM")
    try:
        last = None
        for l in reversed(model.layers):
            if isinstance(l, tf.keras.layers.Conv2D): last = l.name; break
        if last is None:
            for l in reversed(model.layers):
                if hasattr(l,"layers"):
                    for s in reversed(l.layers):
                        if isinstance(s, tf.keras.layers.Conv2D): last=s.name; break
                if last: break
        gm = tf.keras.Model(inputs=model.inputs,
                            outputs=[model.get_layer(last).output, model.output])
        with tf.GradientTape() as tape:
            inp_t = tf.cast(batch, tf.float32)
            co, p2 = gm(inp_t); cc = p2[:, top_idx]
        g = tape.gradient(cc, co)
        pg = tf.reduce_mean(g, axis=(0,1,2))
        hm = (co[0] @ pg[..., tf.newaxis]).numpy().squeeze()
        hm = np.maximum(hm, 0); hm /= (hm.max()+1e-8)
        u8 = (arr*255).astype(np.uint8)
        hr = cv2.resize(hm, (IMG_SIZE[1], IMG_SIZE[0]))
        hc = cv2.applyColorMap(np.uint8(255*hr), cv2.COLORMAP_JET)
        hrgb = cv2.cvtColor(hc, cv2.COLOR_BGR2RGB)
        ov = (0.45*hrgb + 0.55*u8).astype(np.uint8)
        st.image(ov, caption="Grad-CAM overlay", width="content")
    except Exception as e:
        st.warning(f"Grad-CAM unavailable: {e}")
st.divider(); st.caption("TensorFlow · Keras · Streamlit")
