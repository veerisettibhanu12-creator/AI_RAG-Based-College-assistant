from pathlib import Path
import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title='AI RAG College Assistant', page_icon='🎓', layout='wide')
DOCS=Path('data/documents'); DB=Path('data/chroma_db'); DOCS.mkdir(parents=True,exist_ok=True); DB.mkdir(parents=True,exist_ok=True)
@st.cache_resource
def engine(): return RAGEngine(str(DB))
rag=engine()
st.title('🎓 AI RAG Based College Assistant')
st.caption('Ask questions from your college PDF documents.')
with st.sidebar:
    st.header('📚 College Documents')
    files=st.file_uploader('Upload PDF files',type=['pdf'],accept_multiple_files=True)
    if st.button('➕ Add PDFs',use_container_width=True):
        if not files: st.warning('Please select a PDF.')
        else:
            n=0
            for f in files:
                p=DOCS/f.name; p.write_bytes(f.getbuffer()); n += int(rag.add_pdf(p))
            st.success(f'{n} PDF(s) added.')
    if st.button('🗑️ Clear Knowledge Base',use_container_width=True):
        rag.clear_database(); st.session_state.messages=[]; st.rerun()
    st.divider(); st.markdown('**RAG:** PDF → chunks → embeddings → ChromaDB → retrieval → AI answer')
if 'messages' not in st.session_state: st.session_state.messages=[]
for m in st.session_state.messages:
    with st.chat_message(m['role']):
        st.markdown(m['content'])
        if m.get('sources'):
            with st.expander('📖 Sources'):
                for s in m['sources']: st.write(s)
q=st.chat_input('Ask about your college documents...')
if q:
    st.session_state.messages.append({'role':'user','content':q})
    with st.chat_message('user'): st.markdown(q)
    with st.chat_message('assistant'):
        with st.spinner('Retrieving information...'): result=rag.answer(q)
        st.markdown(result['answer'])
        if result['sources']:
            with st.expander('📖 Retrieved Sources'):
                for s in result['sources']: st.write(s)
    st.session_state.messages.append({'role':'assistant','content':result['answer'],'sources':result['sources']})
