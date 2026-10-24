from itertools import count
import json, math, os
import streamlit as st
import generate as G
from model import (
    load_generator,
    load_discriminator,
    find_model_path,
    LATENT_DIM,
)
import tensorflow as tf

st.set_page_config(page_title="DigitForge AI", page_icon="🧠", layout="wide")
# Add this immediately after st.set_page_config(...)
st.markdown("""
<style>
/* Main app background and default text */
.stApp {
    background: linear-gradient(135deg, #0b1020 0%, #111827 55%, #17132b 100%);
    color: #ffffff;
}

.stApp, .stApp p, .stApp label,
.stApp li, .stApp span,
.stApp [data-testid="stMarkdownContainer"] {
    color: #f3f4f6;
}

/* Main headings */
h1, h2, h3, h4,
.stApp h1, .stApp h2, .stApp h3 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #374151;
}

.stTabs [data-baseweb="tab"] {
    color: #e5e7eb !important;
    background: transparent;
    border-radius: 8px 8px 0 0;
    padding: 12px 16px;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important;
    background: #312e81;
    border-bottom: 3px solid #a78bfa;
}

.stTabs [data-baseweb="tab"] p {
    color: inherit !important;
    font-weight: 600;
}

/* Inputs, dropdowns and select sliders */
.stApp [data-baseweb="select"] > div,
.stApp [data-baseweb="input"] > div,
.stApp [data-baseweb="base-input"] {
    background-color: #1f2937 !important;
    border-color: #6b7280 !important;
}

.stApp input,
.stApp textarea,
.stApp [data-baseweb="select"] *,
.stApp [data-testid="stSelectSlider"] * {
    color: #ffffff !important;
}

.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] label {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Buttons */
.stApp .stButton > button,
.stApp button[kind="primary"] {
    background: linear-gradient(90deg, #6d28d9, #4f46e5);
    color: #ffffff !important;
    border: 1px solid #a78bfa;
    border-radius: 10px;
    font-weight: 700;
    padding: 0.55rem 1rem;
    transition: 0.2s ease;
}

.stApp .stButton > button:hover {
    border-color: #ffffff;
    filter: brightness(1.15);
}

/* Metrics and content cards */
.stApp [data-testid="stMetric"] {
    background: #1f2937;
    border: 1px solid #374151;
    padding: 14px;
    border-radius: 12px;
}

.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricLabel"] p {
    color: #d1d5db !important;
}

.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricValue"] * {
    color: #ffffff !important;
}

/* Captions and helper text: brighter than default muted grey */
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] * {
    color: #d1d5db !important;
}

/* Alerts and expanders */
.stApp [data-testid="stAlert"] {
    border-radius: 10px;
}

.stApp [data-testid="stExpander"] {
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 10px;
}

/* Sidebar */
.stApp [data-testid="stSidebar"] {
    background: #0f172a;
}

.stApp [data-testid="stSidebar"] * {
    color: #f9fafb;
}

/* Divider */
.stApp hr {
    border-color: #374151;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
.stApp{background:#0b1026;color:#e6e9f5}
h1,h2,h3{letter-spacing:-.02em}
.hero{padding:1.2rem 1.4rem;border:1px solid #2a3470;border-radius:16px;
 background:linear-gradient(135deg,#141a3d,#1b1450)}
.hero h1{font-size:2.6rem;margin:0 0 .3rem;
 background:linear-gradient(90deg,#a78bfa,#38bdf8);-webkit-background-clip:text;color:transparent}
.badge{display:inline-block;padding:.15rem .6rem;border:1px solid #38bdf8;border-radius:99px;
 color:#7dd3fc;font-size:.8rem}
.card{border:1px solid #2a3470;border-radius:12px;padding:1rem;background:#10173a}
.stButton>button,.stDownloadButton>button{background:#6d4aff;color:#fff;border:0;border-radius:10px}
.stButton>button:hover,.stDownloadButton>button:hover{background:#38bdf8;color:#0b1026}
.stButton>button:focus-visible{outline:2px solid #38bdf8}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Generator…")
def get_model():
    return load_generator()

@st.cache_resource(show_spinner="Loading Discriminator…")
def get_discriminator():
    return load_discriminator()


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


try:
    model = get_model()
    err = None
except Exception as e:  # missing file or incompatible TF
    model, err = None, str(e)

st.markdown("""<div class="hero"><span class="badge">All images are AI-generated</span>
<h1>DigitForge AI</h1>
<p>A DCGAN Generator trained on MNIST turns 100 random numbers into a new 28×28 handwritten digit.
It learns by competing with a Discriminator that tries to tell its images from real ones.</p></div>""",
            unsafe_allow_html=True)

if model is None:
    st.error("The trained Generator could not be loaded.")
    st.code(err)
    st.info("Place `dcgan_generator.h5` in the `artifacts/` folder and restart the app. "
            "If TensorFlow versions differ, re-export from Colab with `generator.save('generator.keras')`.")
    st.stop()

tab_hero, tab_studio, tab_lab, tab_model, tab_train, tab_about = st.tabs(
    [
        "Home",
        "Generation studio",
        "Real vs Generated Lab",
        "Model",
        "Training insights",
        "About",
    ]
)

with tab_hero:
    st.subheader("Fresh from the Generator")
    hero = G.grid_image(G.to_uint8(G.generate(model, 16, seed=7)), cols=8, scale=5)
    st.image(hero, caption="16 digits generated from fixed seed 7 (reproducible preview).")
    st.write("Open **Generation studio** and press **Generate Digits** for new random samples.")

with tab_studio:
    c1, c2 = st.columns([1, 3])
    with c1:
        n = st.select_slider("Number of images", options=[4, 9, 16, 25], value=16)
        if st.button("Generate Digits", use_container_width=True) or "imgs" not in st.session_state:
            st.session_state.imgs = G.to_uint8(G.generate(model, n))
        st.caption("Each click samples new latent vectors z ~ N(0, 1).")
    imgs = st.session_state.imgs
    cols_n = int(math.sqrt(len(imgs))) if int(math.sqrt(len(imgs)))**2 == len(imgs) else 4
    grid = G.grid_image(imgs, cols=cols_n, scale=8)
    with c2:
        st.image(grid, width=600)
        st.download_button("Download full grid (PNG)", G.png_bytes(grid), "digitforge_grid.png", "image/png")
    with st.expander("Download individual images"):
        cc = st.columns(5)
        for i, a in enumerate(imgs):
            im = G.to_pil(a, 8)
            with cc[i % 5]:
                st.image(im, width=90)
                st.download_button("PNG", G.png_bytes(im), f"digit_{i+1:02d}.png", "image/png", key=f"d{i}")

with tab_lab:
    st.subheader("Real vs Generated Lab")
    st.write(
        "Generate handwritten digits and inspect the trained "
        "Discriminator's real-like scores."
    )

    try:
        discriminator = get_discriminator()
    except Exception as e:
        st.error("Could not load the trained Discriminator.")
        st.code(str(e))
    else:
        st.markdown("### Number of digits")
st.caption("Choose how many handwritten digits to generate.")
count = st.select_slider(
    "Number of digits",
    options=[4, 9, 16],
    value=9,
    key="lab_count",
    label_visibility="collapsed")
if st.button("Generate and evaluate", key="evaluate_digits"):
            generated = G.generate(model, count)

            # Discriminator expects image pixels in [-1, 1].
            scores = tf.sigmoid(
                discriminator(generated, training=False)
            ).numpy().reshape(-1)

            images_u8 = G.to_uint8(generated)
            columns = st.columns(3)

            for i, (image, score) in enumerate(zip(images_u8, scores)):
                with columns[i % 3]:
                    st.image(image, width=140)
                    st.metric("Real-like score", f"{score:.1%}")

                    if score >= 0.5:
                        st.caption("Discriminator leans real")
                    else:
                        st.caption("Discriminator leans generated")

            st.caption(
                "These are sigmoid-transformed Discriminator scores, "
                "not calibrated probabilities of authenticity."
            )

with tab_model:
    meta = read_json("artifacts/training_history.json") or {}
    st.table({
        "Property": ["Model type", "Dataset", "Output", "Latent dimension", "Framework",
                     "Loaded file", "Epochs", "Batch size"],
        "Value": ["DCGAN", "MNIST", "28 × 28 grayscale", LATENT_DIM, "TensorFlow / Keras",
                  find_model_path(), meta.get("epochs", "not recorded"),
                  meta.get("batch_size", "not recorded")]})
    st.write("**Generator**: noise → Dense(7·7·256) → 3 transposed-conv blocks → `tanh` image. "
             "**Discriminator**: 2 strided conv blocks with dropout → 1 raw score (used only in training).")
    with st.expander("Generator summary"):
        lines = []
        model.summary(print_fn=lines.append)
        st.code("\n".join(lines))

with tab_train:
    hist = read_json("artifacts/training_history.json")
    if hist and "generator_loss" in hist:
        st.line_chart({"Generator": hist["generator_loss"], "Discriminator": hist["discriminator_loss"]})
    elif os.path.exists("results/training_losses.png"):
        st.image("results/training_losses.png")
    else:
        st.info("No loss history saved yet. Run the training cells in `DigitForge_Training.md` / the "
                "notebook and save `artifacts/training_history.json`.")
    shown = False
    for f in sorted(os.listdir("results")) if os.path.isdir("results") else []:
        if f.startswith("epoch_") and f.endswith(".png"):
            st.image(f"results/{f}", caption=f, width=320); shown = True
    if not shown:
        st.caption("No epoch sample grids found in `results/`.")
    st.caption("GAN losses fluctuate because each network improves against the other; "
               "judge quality by looking at the images, not the loss values.")

with tab_about:
    st.markdown("""
1. **Preprocessing**: MNIST pixels scaled from [0, 255] to [-1, 1] to match the Generator's `tanh` output.
2. **Latent vectors**: 100 values drawn from a standard normal distribution.
3. **Generator**: Dense → Reshape(7×7×256) → Conv2DTranspose ×3 with BatchNorm + LeakyReLU.
4. **Discriminator**: Conv2D ×2 + LeakyReLU + Dropout → one logit.
5. **Training**: binary cross-entropy; D is rewarded for separating real/fake, G for fooling D; Adam, lr 1e-4.
6. **Inference**: only the Generator runs; outputs are mapped back with `x·127.5+127.5`.
7. **Limitations**: the model can't be told which digit to draw (that needs a conditional GAN); some samples are
   malformed, and GANs can suffer mode collapse (repeated outputs).

Based on the reference notebook in *rohanmistry231/Deep-Learning-Projects* (Project 09), which follows the TensorFlow DCGAN tutorial.""")
