from dotenv import load_dotenv
import os
from llama_index.readers import SimpleDirectoryReader
from pinecone import Pinecone
load_dotenv()
import openai
import streamlit as st
from llama_index.vector_stores import PineconeVectorStore
from llama_index import VectorStoreIndex
from llama_index.callbacks import LlamaDebugHandler, CallbackManager
from llama_index import ServiceContext
from llama_index.postprocessor import SentenceEmbeddingOptimizer

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager(handlers=[llama_debug])
service_context = ServiceContext.from_defaults(callback_manager=callback_manager)


from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import load_summarize_chain


def summarize_pdf(pdf_file_path):
    """Summarizes the given PDF using OpenAI API.

    Args:
        pdf_file_path (str): The path to the PDF file.

    Returns:
        str: The summary of the PDF, or None if an error occurs.
    """

    try:
        llm      = OpenAI(temperature=0,api_key=os.environ["OPENAI_APIKEY"])
        loader   = PyPDFLoader(pdf_file_path)

        docs    = loader.load_and_split()
        chain   = load_summarize_chain(llm=llm, chain_type="map_reduce")
        summary = chain.run(docs)   
        return summary
    except Exception as e:
        st.error(f"An error occurred: {e}")
    return None
st.set_page_config(
        page_title="RAG documents",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )

# Divide the window into two columns
col1, col2 = st.columns(2)

# Summarization section in the first column
with col1:
    st.title("PDF Summarization")
    pdf_file_path = st.text_input("Enter PDF File Path:", placeholder="e.g., /path/to/your/pdf.pdf")

    if st.button("Summarize PDF"):
        # Show a progress spinner while summarizing
        with st.spinner("Summarizing..."):
            summary = summarize_pdf(pdf_file_path)

        if summary:
            st.success("Summary:")
            st.write(summary)
        else:
            st.error("An error occurred. Please check the file or try again later.")
@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    pine = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )
    index_name = "llamaindex-documentation-helper"
    pinecone_index = pine.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, service_context=service_context
    )



# RAG system section in the second column
with col2:
    index = get_index()

    if "chat_engine" not in st.session_state.keys():
        postprocessor = SentenceEmbeddingOptimizer(
            embed_model=service_context.embed_model,
            percentile_cutoff=0.5,
            threshold_cutoff=0.7,
        )
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="context", verbose=True,
        )

    
    st.title("Chat with RAG docs")
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask me a question about Retreival Augmented Generation?",
            }
        ]
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking...."):
                response = st.session_state.chat_engine.chat(message=prompt)
                st.write(response.response)
                nodes = [node for node in response.source_nodes]
                for col, node, i in zip(st.columns(len(nodes)), nodes, range(len(nodes))):
                    with col:
                        st.header(f"Source Node  {i+1}: score = {node.score} ")
                        st.write(node.text)
                    message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)