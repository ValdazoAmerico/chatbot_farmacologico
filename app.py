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
import weaviate

@st.cache_resource
def get_chain():
	base_embeddings = OpenAIEmbeddings()
	llm_hyde = OpenAI()
	
	prompt_template = """Redacta un fragmento de un artículo científico para responder a la pregunta.
	Pregunta: {question}
	Respuesta:"""
	prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
	llm_chain = LLMChain(llm=llm_hyde, prompt=prompt)
	
	embeddings = HypotheticalDocumentEmbedder(
	    llm_chain=llm_chain, base_embeddings=base_embeddings
	)
	
	
	auth_config = weaviate.AuthApiKey(api_key=os.environ['WEAVIATE_API_KEY'])
	
	client = weaviate.Client(url=os.environ['WEAVIATE_URL'], auth_client_secret=auth_config, additional_headers={
	        "X-OpenAI-Api-Key": os.environ['OPENAI_API_KEY'], # Replace with your OpenAI key
	        })
	retriever = WeaviateHybridSearchRetriever(
    	client=client,
    	index_name="LangChain",
    	text_key="text",
    	attributes=[],
    	create_schema_if_missing=True,
)
	#vectordb = WeaviateLangChain(client=client,  index_name="ICC", text_key="content", embedding=embeddings)
	
	#retriever = vectordb.as_retriever(search_kwargs={"k": 3})
	
	prompt=PromptTemplate(
	    template="""Actúas como un médico cardiólogo especializado. Tu tarea consiste en proporcionar respuestas precisas y fundamentadas en el campo de la cardiología, basándote únicamente en la información proporcionada en el texto médico que se te presente. Tu objetivo es comportarte como un experto en cardiología y ofrecer asistencia confiable y precisa.

Debes responder solo a preguntas relacionadas con cardiología en función del contexto proporcionado. Estás comprometido a brindar respuestas confiables y basadas en la evidencia médica presentada. En caso de desconocer la respuesta o no contar con información para responder la pregunta, simplemente dirás 'No lo sé'. No intentarás inventar una respuesta.

----------------

{context}

----------------

""",
	    input_variables=["context"],
	)
	system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
	
	prompt=PromptTemplate(
	    template="""{question}""",
	    input_variables=["question"],
	)
	human_message_prompt = HumanMessagePromptTemplate(prompt=prompt)
	
	chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
	
	condense_template = """Dada la siguiente conversación y una pregunta de seguimiento, reformula la pregunta de seguimiento para que sea una pregunta independiente.
	
Conversación:
{chat_history}

Pregunta de seguimiento: {question}

Pregunta independiente:"""
	CONDENSE_QUESTION_PROMPT = PromptTemplate(template=condense_template, input_variables=["chat_history", "question"])
	
	llm = ChatOpenAI()
	
	question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
	
	llm2 = ChatOpenAI(temperature=0, verbose=True)
	llm3 = ChatOpenAI(temperature=0, verbose=True, max_tokens=500)
	question_generator = LLMChain(llm=llm2, prompt=CONDENSE_QUESTION_PROMPT)
	doc_chain = load_qa_chain(llm3, chain_type="stuff", verbose=True)
	
	chain = ConversationalRetrievalChain(
	    retriever=retriever,
	    question_generator=question_generator,
	    combine_docs_chain=doc_chain,
	    verbose=True, return_source_documents=True
	)
	
	chain.combine_docs_chain.llm_chain.prompt = chat_prompt
	return chain
	
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
	
		chain = get_chain()
	
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
		            st.session_state['generated'].append('¡Hola! Soy CardioBot, una herramienta especializada para apoyar a los médicos en el análisis de textos relacionados con cardiología. Mi conocimiento se basa en información basada en evidencia científica sobre tratamientos y medicación en esta área.')
		        else:
		            try:
		                    docs = lotr.get_relevant_documents(user_input)
		                    if len(st.session_state.ai) == 0:
		                        response = chain({"question": user_input, "chat_history": []})
		                        output = response['answer']
		                        #docs = response['source_documents']
		                        raw_string = ''
		                        for d in range(len(docs)):
		                        	raw_string += f'Extracto {d+1}:\n'
		                        	raw_string += docs[d].page_content.replace('\n', ' ')
		                        	raw_string += '\n'
		                        	#raw_string += f"Página {str(docs[d].metadata['page'])}"
		                        	raw_string += '\n\n'
		                        print(docs)
		                        print("DOCS")
		                        st.session_state['data'].append(raw_string)
		                        st.session_state.ai.append(output)
		                        st.session_state.past.append(user_input)
		                        st.session_state['generated'].append(output)   
		                    elif len(st.session_state.ai) == 1:
		                        chat_history = [(st.session_state['past'][-1], st.session_state['generated'][-1])]
		                        response = chain({"question": user_input, "chat_history": chat_history})
		                        output = response['answer']
		                        #docs = response['source_documents']   
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
		                    else:
		                        chat_history = [(st.session_state['past'][-2], st.session_state['generated'][-2]), (st.session_state['past'][-1], st.session_state['generated'][-1])]                
		                        response = chain({"question": user_input, "chat_history": chat_history})
		                        output = response['answer']
		                        #docs = response['source_documents']
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
	

		if st.session_state['data']:
		        patient_data = st.session_state['data'][-1]
		else:
			patient_data = " "
		st.subheader("Información de contexto:")
		stx.scrollableTextbox(patient_data)
