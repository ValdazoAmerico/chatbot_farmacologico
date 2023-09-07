import streamlit as st
from streamlit_chat import message
import streamlit_scrollable_textbox as stx
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS, Weaviate as WeaviateLangChain
import logging
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks import get_openai_callback
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain.chains import ConversationChain

import weaviate

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

auth_config = weaviate.AuthApiKey(api_key=os.environ['WEAVIATE_API_KEY'])
	
client = weaviate.Client(url=os.environ['WEAVIATE_URL'], auth_client_secret=auth_config, additional_headers={
	        "X-OpenAI-Api-Key": os.environ['OPENAI_API_KEY'], # Replace with your OpenAI key
	        })
auth_config2 = weaviate.AuthApiKey(api_key=os.environ['WEAVIATE_API_KEY2'])
	
client2 = weaviate.Client(url=os.environ['WEAVIATE_URL2'], auth_client_secret=auth_config2, additional_headers={
	        "X-OpenAI-Api-Key": os.environ['OPENAI_API_KEY'], # Replace with your OpenAI key
	        })

retriever = WeaviateHybridSearchRetriever(
    	client=client2,
    	index_name="LangChain",
    	text_key="text",
    	attributes=[]
)

def answer_question(question, history):
      query_result = (
      client.query
      .get("Evicardio", ["content", "keywords"])
      .with_bm25(question)
      .with_limit(4)
      .do()
  )
      res = retriever.get_relevant_documents(question)

      arr=[]
      for r in res:
        arr.append(r.page_content)

      for r in query_result['data']['Get']['Evicardio']:
        arr.append(r['content'])
      print("len", len("\n\n".join(arr)))
      return ask("\n\n".join(arr),question, history), "\n\n".join(arr)

def ask(context, question, history):

	default_template = f"""Actúa como un médico cardiólogo especializado. Tu tarea consiste en proporcionar respuestas precisas y fundamentadas en el campo de la cardiología, basándote únicamente en la información proporcionada en el texto médico que se te presente. Tu objetivo es comportarte como un experto en cardiología y ofrecer asistencia confiable y precisa.

Debes responder solo a preguntas relacionadas con cardiología en función del contexto proporcionado. Estás comprometido a brindar respuestas confiables y basadas en la evidencia médica presentada. Mantén la respuesta breve y concisa. En caso de desconocer la respuesta o no contar con información para responder la pregunta, simplemente dirás 'No lo sé'. No intentarás inventar una respuesta.

----------------

{context}

----------------

"""	
	chat_template = default_template + """{history}
{input}"""

	prompt = PromptTemplate(input_variables=["history", "input"], template=chat_template)
	chat = ChatOpenAI(temperature=0, verbose=True)
	#memory = ConversationBufferWindowMemory(k=2)
	#if len(st.session_state.ai) == 1:
		#memory.save_context({"input": st.session_state.past[-1]}, {"output": st.session_state.ai[0]})
	#if len(st.session_state.ai) > 1:
		#memory.save_context({"input": st.session_state.past[-2]}, {"output": st.session_state.ai[-2]})
		#memory.save_context({"input": st.session_state.past[-1]}, {"output": st.session_state.ai[-1]})

	print("history",history)
	try:
		conversation = ConversationChain(
			llm=chat,
			verbose=False,
			prompt=prompt)
		if history:
			response = conversation.predict(input=question, history=chat_history)
		else:
			response = conversation.predict(input=question)
	except:
			print("16k tokens")
			chat = ChatOpenAI(temperature=0, verbose=True, model='gpt-3.5-turbo-16k')
			conversation = ConversationChain(
				llm=chat,
				verbose=False,
				prompt=prompt)
			if history:
				response = conversation.predict(input=question, history=chat_history)
				print("response history", response)
			else:
				response = conversation.predict(input=question)
				print("response NO history", response)


	return response

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == 'uma2023':
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():
	st.title("CardioBot :hospital:")
	col1, col2 = st.columns(2)
	with col1:
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
		            st.session_state['generated'].append('¡Hola! Soy CardioBot, una herramienta especializada para apoyar a los médicos en el análisis de textos relacionados con cardiología. Mi conocimiento se basa en información basada en evidencia científica sobre tratamientos y medicación en esta área.')
		        else:
		            try:
		                    #docs = retriever.get_relevant_documents(user_input)
		                    if len(st.session_state.ai) == 0:
		                        print("here 1")
		                        output, raw_string = answer_question(user_input, [])
		                        #response = chain({"question": user_input, "chat_history": []})
		                        # output = response['answer']
		                        # docs = response['source_documents']
		                        # raw_string = ''
		                        # for d in range(len(docs)):
		                        # 	raw_string += f'Extracto {d+1}:\n'
		                        # 	raw_string += docs[d].page_content.replace('\n', ' ')
		                        # 	raw_string += '\n'
		                        # 	#raw_string += f"Página {str(docs[d].metadata['page'])}"
		                        # 	raw_string += '\n\n'
		                        st.session_state['data'].append(raw_string)
		                        st.session_state.ai.append(output)
		                        st.session_state.past.append(user_input)
		                        st.session_state['generated'].append(output)   
		                    elif len(st.session_state.ai) == 1:
		                        print("here 2")
		                        chat_history = [(st.session_state['past'][-1], st.session_state['generated'][-1])]
		                        output, raw_string = answer_question(user_input, chat_history)
		                        #response = chain({"question": user_input, "chat_history": chat_history})
		                        # output = response['answer']
		                        # docs = response['source_documents']   
		                        # raw_string = ''
		                        # for d in range(len(docs)):
		                        # 	raw_string += f'Extracto {d+1}:\n'
		                        # 	raw_string += docs[d].page_content.replace('\n', ' ')
		                        # 	raw_string += '\n'
		                        # 	#raw_string += f"Página {str(docs[d].metadata['page'])}"
		                        # 	raw_string += '\n\n'
		                        st.session_state['data'].append(raw_string) 
		                        st.session_state.ai.append(output)
		                        st.session_state.past.append(user_input)
		                        st.session_state['generated'].append(output)
		                    else:
		                        chat_history = [(st.session_state['past'][-2], st.session_state['generated'][-2]), (st.session_state['past'][-1], st.session_state['generated'][-1])]                
		                        response = chain({"question": user_input, "chat_history": chat_history})
		                        output = response['answer']
		                        docs = response['source_documents']
		                        raw_string = ''
		                        for d in range(len(docs)):
		                        	raw_string += f'Extracto {d+1}:\n'
		                        	raw_string += docs[d].page_content.replace('\n', ' ')
		                        	raw_string += '\n'
		                        	#raw_string += f"Página {str(docs[d].metadata['page'])}"
		                        	raw_string += '\n\n'
		                        st.session_state['data'].append(raw_string)
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
		if not st.session_state['data']:
			patient_data = "‎ "
		else:
			patient_data = st.session_state['data'][-1]
		if patient_data:
			st.subheader("Información de contexto:")
			stx.scrollableTextbox(patient_data,height = 350)
