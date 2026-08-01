import streamlit as st
import joblib

# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(
    page_title="Fake Product Review Detector",
    page_icon="🛒",
    layout="centered"
) 
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: black(135deg, #E3F2FD, #BBDEFB);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model.joblib")

# -----------------------------
# Title
# -----------------------------
st.markdown(
    "<h1 style='text-align:center; color:#4CAF50;'>🛒 Fake Product Review Detection</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Detect whether a product review is <b>Fake</b> or <b>Genuine</b> using Machine Learning.</p>",
    unsafe_allow_html=True
)

st.divider()

st.info("💡 Enter a product review below and click **Check Review**.")

# -----------------------------
# User Input
# -----------------------------
review = st.text_area(
    "📝 Product Review",
    height=180,
    placeholder="Type or paste your product review here..."
)

# -----------------------------
# Buttons
# -----------------------------
col1, col2 = st.columns(2)

check = col1.button("🔍 Check Review", use_container_width=True)
reset = col2.button("🔄 Reset", use_container_width=True)

if reset:
    st.rerun()

# -----------------------------
# Prediction
# -----------------------------
if check:

    if review.strip() == "":
        st.warning("⚠️ Please enter a review.")

    else:

        with st.spinner("🤖 AI is analyzing the review..."):

            prediction = model.predict([review])
            probability = model.predict_proba([review])[0]
            confidence = max(probability) * 100

        st.divider()

        st.subheader("📝 Review Entered")
        st.success(review)

        st.subheader("📊 Prediction Result")

        if prediction[0] == 1:
            st.error("🚨 Fake Review Detected")
            st.snow()
        else:
            st.success("✅ Genuine Review")
            st.balloons()

        st.write(f"🎯 Confidence Score: **{confidence:.2f}%**")

        st.progress(int(confidence))

st.divider()

st.caption("🤖 Built with Streamlit & Scikit-learn")