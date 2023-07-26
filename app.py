import streamlit as st
from streamlit_chat import message
import streamlit_scrollable_textbox as stx

from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

if 'generated' not in st.session_state:
        st.session_state['generated'] = []

if 'past' not in st.session_state:
        st.session_state['past'] = []
        
if 'ai' not in st.session_state:
        st.session_state['ai'] = []

embeddings = OpenAIEmbeddings()


vectordb = FAISS.load_local('./faiss_index/', embeddings)
llm = ChatOpenAI(temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm
)

condense_template = """Dada la siguiente conversación y una pregunta de seguimiento, reformula la pregunta de seguimiento para que sea una pregunta independiente.

Historial del chat:
{chat_history}
Entrada de seguimiento: {question}
Pregunta independiente:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate(template=condense_template, input_variables=["chat_history", "question"])
llm1 = OpenAI()
question_generator = LLMChain(llm=llm1, prompt=CONDENSE_QUESTION_PROMPT)
question_prompt_template = """Utiliza el siguiente extracto de un largo documento para ver si algún texto es relevante para responder la pregunta.

Extracto: {context}
Pregunta: {question}
Texto relevante, si lo hay:"""
QUESTION_PROMPT = PromptTemplate(
        template=question_prompt_template, input_variables=["context", "question"]
    )

combine_prompt_template = """Dados los siguientes extractos de un largo documento y una pregunta, crea una respuesta final. Si no hay texto relevante para responder la pregunta no trates de inventar una respuesta, di simplemente que 'No hay texto relevante para responder la pregunta'.

{summaries}

Human: {question}
Assistant:"""
    
COMBINE_PROMPT = PromptTemplate(
        template=combine_prompt_template, input_variables=["summaries", "question"]
    )
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)
if len(st.session_state.ai) == 1:
		memory.save_context({"question": st.session_state.past[-1]}, {"output": st.session_state.ai[0]})
if len(st.session_state.ai) > 1:
		memory.save_context({"question": st.session_state.past[-2]}, {"output": st.session_state.ai[-2]})
		memory.save_context({"question": st.session_state.past[-1]}, {"output": st.session_state.ai[-1]})
qa = load_qa_chain(OpenAI(temperature=0), chain_type="map_reduce", question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT, verbose=True)
chain = ConversationalRetrievalChain(retriever=retriever_from_llm,return_source_documents=False, combine_docs_chain=qa, question_generator=question_generator, verbose=True, memory=memory)


st.title("PharmaAssistant 👩‍⚕️")


def clear_text():
	st.session_state["temp"] = st.session_state["text"]
	st.session_state["text"] = ""
		
		
def get_text():
	input_text = st.text_input("You: ", "", key="text",on_change=clear_text)
	st.session_state['user_input'].append(input_text)
	if st.session_state['temp'] == "":
		return "Hola!"
	else:
		return st.session_state['temp']

user_input = get_text()

if user_input:
        if user_input == 'Hola!':
            st.session_state['past'] = []
            st.session_state['generated'] = []
            st.session_state.past.append("Hola")
            st.session_state['generated'].append('¡Hola! Hacé tu consulta sobre tratamientos farmacológicos.')
        else:
            try:
                    print("QUESTION", user_input)
                    output = chain({"question":user_input})['answer']
                    st.session_state.ai.append(output)
                    st.session_state.past.append(user_input)
                    st.session_state['generated'].append(output)
            except:
                pass

if st.session_state['generated']:
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


	
