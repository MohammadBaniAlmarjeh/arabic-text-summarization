from urllib.parse import unquote

import arabic_reshaper
import streamlit as st
from bidi.algorithm import get_display

st.set_page_config(
    page_title="Arabic Text Summarization",
    page_icon="📖",
    initial_sidebar_state="expanded"
    # layout="wide"
)

from summarize import get_results

rtl = lambda w: get_display(f"{arabic_reshaper.reshape(w)}")

st.title("تَلْخِيصُ اَلنُّصُوصِ بِاللُّغَةِ اَلْعَرَبِيَّةِ")

st.markdown(
    """
<style>
@import url(https://fonts.googleapis.com/earlyaccess/scheherazade.css);
section.main {
    background-color: beige;
}
.stMarkdown h1, .main .element-container.css-o7ulmj.e1tzin5v3 {
    text-align: right;
}
.stMarkdown div.css-nlntq9.e16nr0p33 {
    font-weight: bold;
}
textarea {
    direction: rtl;
    height: 140px;
}
.stTextArea .css-qrbaxs {
    float: right;
    font-size: 23px;
}
h1 {
    font-family: 'Scheherazade', serif;
}

.main div.css-nlntq9.e16nr0p33 > p {
    direction: rtl;
}
.main .stMarkdown div.css-nlntq9 p {
    font-size: 22px;
}
.main .stMarkdown div.css-nlntq9 {
    direction: rtl;
}
.main p, .main div, .main input, .main label {
  text-align: right;
  direction: rtl;
}
.main  div>h1>div {
    left: 0;
}
.main button {
    font-size: 22px;
}

</style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.image("svu.png", width=150)
st.sidebar.write("\n")
st.sidebar.write("Arabic Text Summarization")
st.sidebar.write("Made by Mohammad Bani Almarjeh ([Linkedin](https://www.linkedin.com/in/mohammad-bani-almarjeh))")
st.sidebar.write("\n")

model_selected = st.sidebar.selectbox(
     'Select a model',
     ('T5','BERT2BERT', 'GPT-2', 'mBERT2mBERT','Transformer'))
st.sidebar.write("\n")
num_beams = st.sidebar.slider(
    "Number of beams", min_value=1, max_value=10, value=3, step=1
)

length_pe_slider_disabled = False
if model_selected == "GPT-2":
    length_pe_slider_disabled = True

st.sidebar.write("\n")
length_penalty = st.sidebar.slider(
    "Length penalty ", min_value=0.1, max_value=3.0, value=1.0, step=0.1, disabled=length_pe_slider_disabled
)

txt = """يجري علماء في بريطانيا تجربة لاختبار فعالية عقار إيبوبروفين لمساعدة المصابين بفيروس كورونا. وذكرت هيئة الإذاعة البريطانية "بي بي سي" أن فريق مشترك من أطباء مستشفيات "جاي" و"سانت توماس" و"كينغز كوليدج" في لندن يعتقد أن إيبوبروفين، وهو مضاد للالتهابات ومسكن للألم، يمكن أن يعالج صعوبات التنفس.
ويأمل العلماء أن يساعد هذا العلاج المنخفض التكلفة المرضى في الاستغناء عن أجهزة التنفس الصناعي. وذكرت أنه خلال فترة الاختبار، سيحصل نصف المرضى على إيبوبروفين بالإضافة إلى الرعاية المعتادة، حيث سيتم استخدام تركيبة خاصة من إيبوبروفين بدلا من الأقراص العادية التي قد يشتريها الناس عادة."""
text = st.text_area("أدخل نص ليتم تلخيصه", value=txt)

run_query = st.button("لخّص")
if run_query:
    # https://discuss.streamlit.io/t/showing-a-gif-while-st-spinner-runs/5084
    with st.spinner("جاري التلخيص ..."):
        result = get_results(text, model_selected, num_beams, length_penalty)
    if len(result) > 0:
        st.write(result)
    else:
        st.write("")
