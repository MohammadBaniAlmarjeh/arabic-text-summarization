from urllib.parse import unquote

import arabic_reshaper
import streamlit as st
from bidi.algorithm import get_display

from summarize import get_results

st.set_page_config(
    page_title="Arabic Text Summarization",
    page_icon="📖",
    initial_sidebar_state="expanded"
    # layout="wide"
)

rtl = lambda w: get_display(f"{arabic_reshaper.reshape(w)}")


_, col1, _ = st.columns(3)

with col1:
    st.image("svu.png", width=200)
    st.title("تلخيص النصوص باللغة العربية")

st.markdown(
    """
<style>
p, div, input, label {
  text-align: right;
}
</style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Info")
st.sidebar.write("Made by Mohammad Bani Almarjeh")
st.sidebar.image("svu.png", width=150)
st.sidebar.write(
    "Source Code [GitHub](https://github.com/)"
)
st.sidebar.write("\n")
n_answers = st.sidebar.slider(
    "Max. number of answers", min_value=1, max_value=10, value=2, step=1
)

text = st.text_area("", value="محفزات ومزايا وإجراءات كثيرة قدمتها الجهات المعنية لتشجيع الاستثمارات في مجال الطاقات المتجددة كإحداث صندوق تمويل مشاريع الطاقات البديلة وقانون الاستثمار رقم 18 للعام 2021 ما شجع على إقبال العديد من المستثمرين على هذه المشاريع في ظل الحاجة الملحة لمصادر إضافية بديلة متجددة تدعم المنظومة الكهربائية في سورية.")

run_query = st.button("أجب")
if run_query:
    # https://discuss.streamlit.io/t/showing-a-gif-while-st-spinner-runs/5084
    with st.spinner("جاري التلخيص ..."):
        result = get_results(text, model=2)

    if len(result) > 0:
        st.write("## :الأجابات هي")
        st.write(result)
        #f"[**المصدر**](<{result['link']}>)"
    else:
        st.write("## 😞 ليس لدي جواب")
