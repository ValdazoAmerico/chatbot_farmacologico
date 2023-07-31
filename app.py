import streamlit as st
from streamlit_chat import message
import streamlit_scrollable_textbox as stx

from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import logging
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings


if 'generated' not in st.session_state:
        st.session_state['generated'] = []

if 'past' not in st.session_state:
        st.session_state['past'] = []
        
if 'ai' not in st.session_state:
        st.session_state['ai'] = []

if "temp" not in st.session_state:
	st.session_state["temp"] = ""

if 'data' not in st.session_state:
	st.session_state['data'] = []
	
embeddings = OpenAIEmbeddings()


vectordb = FAISS.load_local('./faiss_index/', embeddings)
llm = ChatOpenAI(temperature=0)

memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key="question", return_messages=True, k=2)
if len(st.session_state.ai) == 1:
		memory.save_context({"question": st.session_state.past[-1]}, {"output": st.session_state.ai[0]})
if len(st.session_state.ai) > 1:
		memory.save_context({"question": st.session_state.past[-2]}, {"output": st.session_state.ai[-2]})
		memory.save_context({"question": st.session_state.past[-1]}, {"output": st.session_state.ai[-1]})
chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0),
                                   retriever=vectordb.as_retriever(), memory=memory)

col1, col2 = st.columns(2)
with col1:
	st.title("CardioBot :hospital:")
	def clear_text():
		st.session_state["temp"] = st.session_state["text"]
		st.session_state["text"] = ""
			
			
	def get_text():
		input_text = st.text_input("You: ", "", key="text",on_change=clear_text)
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
	            st.session_state['generated'].append('¡Hola! Hacé tu consulta sobre tratamientos farmacológicos para ICC e Hipertensión Arterial Pulmonar.')
	        else:
	            try:
	                    docs = vectordb.as_retriever().get_relevant_documents(user_input)
	                    raw_string = ''
	                    for d in docs:
	                    	raw_string += d.page_content.replace('\n', ' ')
	                    	raw_string += '\n'
	                    st.session_state.data.append(raw_string)
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

with col2:
		st.title("EHR Patient Data")
		if not st.session_state.data:
			patient_data = ""
		else:
			patient_data = st.session_state.data[-1]

		# Display the EHR patient data
		if patient_data:
			st.subheader(f"Context Information:")
			stx.scrollableTextbox(patient_data,height = 350)
