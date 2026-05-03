import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Space Debris Detection",
    page_icon="🛰️",
    layout="wide"
)

# =====================================================
# PRO SPACE UI
# =====================================================
st.markdown("""
<style>

.stApp{
background:
linear-gradient(rgba(2,6,23,0.88), rgba(2,6,23,0.88)),
url("https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&w=1800&q=80");
background-size:cover;
background-position:center;
background-attachment:fixed;
color:white;
}

section[data-testid="stSidebar"]{
background:rgba(3,17,31,0.92);
backdrop-filter: blur(10px);
}

h1,h2,h3,h4,h5,p,label{
color:white !important;
}

.glass{
background:rgba(255,255,255,0.08);
backdrop-filter:blur(10px);
padding:18px;
border-radius:20px;
border:1px solid rgba(255,255,255,0.10);
box-shadow:0 8px 25px rgba(0,0,0,0.28);
}

.metric{
background:linear-gradient(135deg,#06b6d4,#2563eb,#1d4ed8);
padding:18px;
border-radius:18px;
text-align:center;
box-shadow:0 8px 20px rgba(0,0,0,0.28);
}

.small{
font-size:14px;
opacity:0.9;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================
st.markdown("""
<div style='text-align:center;padding:10px'>
<h1>🛰️ AI Space Debris Detection Dashboard</h1>
<p>Real-Time Orbital Threat Monitoring & Smart Space Analysis</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🛠 Mission Control")

uploaded = st.sidebar.file_uploader(
    "Upload Space Image",
    type=["jpg","jpeg","png"]
)

# =====================================================
# LOAD IMAGE
# =====================================================
if uploaded:
    image = Image.open(uploaded).convert("RGB")
else:
    try:
    image = Image.open("images/space.jpg").convert("RGB")
except:
    image = Image.new("RGB", (900, 500), (5, 10, 25))

img = np.array(image)
draw = img.copy()

# =====================================================
# DETECTION
# =====================================================
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

_, thresh = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY)

kernel = np.ones((3,3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
thresh = cv2.dilate(thresh, kernel, iterations=1)

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

rows = []

debris = 0
satellite = 0
planet = 0

H,W,_ = img.shape
img_area = H*W

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area < 40:
        continue

    x,y,w,h = cv2.boundingRect(cnt)
    box_area = w*h
    ratio = box_area / img_area

    if ratio > 0.10 and abs(w-h) < 120:
        label = "Planet"
        color = (0,255,120)
        planet += 1

    elif box_area > 2500:
        label = "Satellite"
        color = (255,255,0)
        satellite += 1

    else:
        label = "Space Debris"
        color = (255,90,90)
        debris += 1

    cv2.rectangle(draw,(x,y),(x+w,y+h),color,2)

    cv2.putText(
        draw,
        label,
        (x,y-8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )

# =====================================================
# SMART REPORT DATA
# =====================================================
rows = [
    {
        "Object Type":"Space Debris",
        "Detected Count":debris,
        "Threat Level":"High" if debris >= 8 else "Medium" if debris >= 4 else "Low"
    },
    {
        "Object Type":"Satellite",
        "Detected Count":satellite,
        "Threat Level":"Safe"
    },
    {
        "Object Type":"Planet",
        "Detected Count":planet,
        "Threat Level":"None"
    }
]

# =====================================================
# METRICS
# =====================================================
m1,m2,m3,m4 = st.columns(4)

with m1:
    st.markdown(f"<div class='metric'><h4>Total Objects</h4><h2>{debris+satellite+planet}</h2></div>", unsafe_allow_html=True)

with m2:
    st.markdown(f"<div class='metric'><h4>Debris</h4><h2>{debris}</h2></div>", unsafe_allow_html=True)

with m3:
    st.markdown(f"<div class='metric'><h4>Satellite</h4><h2>{satellite}</h2></div>", unsafe_allow_html=True)

with m4:
    st.markdown(f"<div class='metric'><h4>Planet</h4><h2>{planet}</h2></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# IMAGES
# =====================================================
left,right = st.columns(2)

with left:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("🌌 Original Space Image")
    st.image(img, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("🤖 Detection Output")
    st.image(draw, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# REPORT + GRAPH
# =====================================================
c1,c2 = st.columns([1.4,1])

with c1:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📋 Detection Summary Report")

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=250)

    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📊 Object Distribution")

    chart = pd.DataFrame({
        "Type":["Debris","Satellite","Planet"],
        "Count":[debris,satellite,planet]
    })

    st.bar_chart(chart.set_index("Type"))
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# ALERT
# =====================================================
st.markdown("<br>", unsafe_allow_html=True)

if debris >= 10:
    st.error("🚨 HIGH RISK : Dense debris field detected.")

elif debris >= 5:
    st.warning("⚠️ MEDIUM RISK : Moderate orbital hazard.")

else:
    st.success("✅ LOW RISK : Safe orbital zone.")

# =====================================================
# LIVE STATUS
# =====================================================
s1,s2,s3 = st.columns(3)

with s1:
    st.markdown("<div class='glass'><h4>📡 Radar</h4><p class='small'>Tracking Active</p></div>", unsafe_allow_html=True)

with s2:
    st.markdown("<div class='glass'><h4>🧠 AI Engine</h4><p class='small'>Operational</p></div>", unsafe_allow_html=True)

with s3:
    st.markdown(f"<div class='glass'><h4>⏱ Last Scan</h4><p class='small'>{time.strftime('%H:%M:%S')}</p></div>", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("<br>", unsafe_allow_html=True)
st.caption("AI Space Debris Detection | Professional Final Dashboard")
