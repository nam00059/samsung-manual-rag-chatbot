# ===== mac fork-safety guard (unchanged) ============================
import os, multiprocessing as mp
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
})
mp.set_start_method("spawn", force=True)
import faiss, streamlit as st
faiss.omp_set_num_threads(1)
# ===================================================================

from rag_backend import rag_answer    # now points to TinyLlama version

# ---------- Streamlit UI -------------------------------------------
st.set_page_config(page_title="Samsung Manual QA (local RAG)", layout="wide")
st.title("📖 Samsung Manual Question-Answering (TinyLlama, CPU-only)")
st.markdown(
    "Ask any question about the **Galaxy S24** user guide. "
    "Everything runs locally – no external APIs."
)

query = st.text_input(
    "Your question",
    placeholder="e.g. How do I enable power saving mode?"
)
k = st.slider("Number of chunks to retrieve (k)", 3, 10, 5, 1)

if st.button("Answer") and query:
    with st.spinner("Thinking…"):
        st.write(rag_answer(query))
