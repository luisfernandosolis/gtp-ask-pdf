#LangChain Libraries
import langchain
langchain.verbose = False
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain import PromptTemplate

##streamlit libraries
import streamlit as st

from streamlit_chat import message

#PdfReader Libraries
from PyPDF2 import PdfReader

##OS library
import os


##Page Configuration
st.set_page_config(page_title="chatbot con PDF", layout="wide")
st.markdown("""<style>.block-container {padding-top: 1rem;}</style>""", unsafe_allow_html=True)

## Sidebar Configuration
st.sidebar.markdown("<h1 style='text-align: center; color: #176B87;'>Cargar Archivo PDF</h1>", unsafe_allow_html=True)
st.sidebar.write("Carga el archivo .pdf con el cual quieres interactuar")
pdf_doc = st.sidebar.file_uploader("", type="pdf")
st.sidebar.write("---")
clear_button = st.sidebar.button("Limpiar Conversación", key="clear")


## Set OpenAI API KEY
OPENAI_API_KEY="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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


if 'responses' not in st.session_state:
    st.session_state['responses'] = ["!Hola!, ¿En qué puedo ayudarte?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []


if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)



def create_embeddings(pdf):
     # Extrayendo texto del pdf
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # Dividiendo en trosos el texto extraído del pdf
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)


      embeddings = OpenAIEmbeddings()


      knowledge_base = FAISS.from_texts(chunks, embeddings)


      return knowledge_base
# create embeddings 
knowledge_base=create_embeddings(pdf_doc)

# CHAT SECTION
st.markdown("<h2 style='text-align: center; color: #176B87;text-decoration: underline;'><strong>Interactúa con el BOT sobre tu documento</strong></h2>", unsafe_allow_html=True)
st.write("---")
# container del chat history
response_container = st.container()
# container del text box
textcontainer = st.container()

##template de la respuesta
prompt_template = """Responda la pregunta con la mayor precisión posible utilizando el contexto proporcionado. si la respuesta no está 
                    contenida en el contexto, digamos "La pregunta está fuera de contexto, 'no me enseñaron ello ☹️' " \n\n
                    contexto: \n {context}?\n
                    pregunta: \n {question} \n
                    respuesta:
                  """
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)



with textcontainer:
    #Formulario del text input
    with st.form(key='my_form', clear_on_submit=True):
        query = st.text_area("Tu:", key='input', height=100)
        submit_button = st.form_submit_button(label='Enviar')

    if query:
        with st.spinner("escribiendo..."):

            #cosine similarity with API Word Embeddings
            docs = knowledge_base.similarity_search(query)

            # respuesta: 4 posibles respuesta
            

            llm = OpenAI(model_name="text-davinci-003")
            chain = load_qa_chain(llm, chain_type="stuff",prompt=prompt)



            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            

        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 

## Consultando chat History para imprimir las respuestas
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i), avatar_style="pixel-art")
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
