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
from langchain.retrievers.merger_retriever import MergerRetriever
import weaviate
from unidecode import unidecode
from langchain.schema.retriever import BaseRetriever, Document
import re
import requests
import json
url = "https://script.google.com/macros/s/AKfycbxrStUsQkyv7oQrEIhWTmT2mSCGrZ6N3SRtjw41YzvoE2GV6E4ZMR43kBVxT_KIoYmMCA/exec"

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
replacement_list = [{'TAVI': 'reemplazo valvular percutaneo'},
                    {'IAMCEST': 'IAM con elevacion del ST'},
                    {'IAMSEST': 'IAM sin elevacion del ST'},
                    {'AI': 'angina inestable'},
                    {'SK': 'estreptoquinasa'},
                    {'DOACs': 'ACOD'},
                    {'DOAC': 'ACOD'},
                    {'NOAC': 'ACOD'},
                    {'FA': 'fibrilación auricular'},
                    {'AA': 'aleteo auricular'},
                    {'ECG': 'electrocardiograma'},
                    {'CDI': 'cardiodefibrilador implantable'},
                    {'TRC': 'terapia de resincronización cardiaca'},
                    {'TRH': 'terapia de reemplazo hormonal'},
                    {'SCA': 'sindrome coronario agudo'},
                    {'AVK': 'antagonistas de la vitamina K'},
                    {'ATC': 'angioplastia coronaria'},
                    {'ICC': 'insuficiencia cardiaca congestiva'},
                    {'ICAD': 'insuficiencia cardiaca aguda'},
                    {'AAA': 'aneurisma de aorta abdominal'},
                    {'EAo': 'estenosis aortica'},
                    {'IAo': 'insuficiencia aortica'},
                    {'TVP': 'trombosis venosa profunda'},
                    {'TEP': 'tromboembolismo pulmonar'},
                    {'RMN': 'resonancia magnetica'}]
stopw = ['tendria',
 'les',
 'fueramos',
 'sintiendo',
 'teniais',
 'hubieron',
 'habeis',
 'al',
 'hubieramos',
 'seas',
 'sereis',
 'ni',
 'estados',
 'o',
 'fueses',
 'habre',
 'nuestras',
 'hubiste',
 'hubiera',
 'habriais',
 'haya',
 'esas',
 'sentidos',
 'tengas',
 'con',
 'estara',
 'era',
 'tendra',
 'os',
 'habiamos',
 'mis',
 'hayan',
 'sois',
 'sobre',
 'teniendo',
 'fui',
 'sera',
 'tenida',
 'tengais',
 'tendriais',
 'otros',
 'estaremos',
 'somos',
 'fuesen',
 'que',
 'estamos',
 'ti',
 'mi',
 'ese',
 'seran',
 'sentida',
 'hubierais',
 'habrian',
 'estada',
 'mi',
 'estan',
 'estoy',
 'como',
 'soy',
 'el',
 'esta',
 'tuyos',
 'vuestras',
 'esteis',
 'tened',
 'esten',
 'tuvo',
 'tuvieron',
 'tenidos',
 'hubimos',
 'fuera',
 'serias',
 'fuesemos',
 'habian',
 'habriamos',
 'teniamos',
 'es',
 'mios',
 'nos',
 'tuve',
 'tendras',
 'estadas',
 'tienes',
 'habido',
 'habiendo',
 'teneis',
 'estareis',
 'otra',
 'nuestra',
 'uno',
 'tu',
 'sus',
 'ante',
 'estaria',
 'mucho',
 'contra',
 'estais',
 'que',
 'tuyo',
 'estando',
 'habrias',
 'estaran',
 'habida',
 'me',
 'sentid',
 'quienes',
 'nada',
 'estuvierais',
 'hubieran',
 'tiene',
 'estuvimos',
 'estarias',
 'eramos',
 'tenia',
 'estariais',
 'habidas',
 'para',
 'estar',
 'esos',
 'suyas',
 'este',
 'quien',
 'tendre',
 'tuya',
 'seras',
 'del',
 'le',
 'tuviesemos',
 'muy',
 'estuvieseis',
 'hemos',
 'hubo',
 'cual',
 'esta',
 'habras',
 'estuvieramos',
 'tenga',
 'hubiesen',
 'tengamos',
 'estuve',
 'tuviera',
 'en',
 'tenido',
 'fuese',
 'poco',
 'hubieseis',
 'las',
 'estuviesemos',
 'esto',
 'tuvieses',
 'tuvieramos',
 'sere',
 'habran',
 'habreis',
 'estes',
 'estarian',
 'seria',
 'durante',
 'tengan',
 'fueron',
 'vuestro',
 'estuvieses',
 'esa',
 'se',
 'mias',
 'hayamos',
 'seriais',
 'estaban',
 'algunas',
 'fuimos',
 'tengo',
 'eras',
 'has',
 'tienen',
 'tendriamos',
 'siente',
 'tus',
 'mas',
 'erais',
 'tuvierais',
 'hayais',
 'nosotros',
 'eso',
 'hubiese',
 'tuyas',
 'habiais',
 'otro',
 'tuvieran',
 'e',
 'mio',
 'habra',
 'una',
 'fuiste',
 'todo',
 'estado',
 'habria',
 'habidos',
 'han',
 'ella',
 'un',
 'eran',
 'seais',
 'tendrian',
 'he',
 'tuvieras',
 'tuviese',
 'fuerais',
 'nosotras',
 'fueran',
 'hubiesemos',
 'cuando',
 'ellos',
 'sin',
 'lo',
 'este',
 'unos',
 'desde',
 'tendremos',
 'si',
 'tendrias',
 'donde',
 'estuvo',
 'tenemos',
 'yo',
 'estemos',
 'tanto',
 'hay',
 'tenias',
 'estuviese',
 'mia',
 'hubisteis',
 'estuvieras',
 'serian',
 'de',
 'estaras',
 'nuestros',
 'no',
 'habremos',
 'la',
 'tuviste',
 'hasta',
 'nuestro',
 'estuviesen',
 'suyos',
 'seriamos',
 'tuvimos',
 'porque',
 'sentidas',
 'fueseis',
 'pero',
 'antes',
 'sean',
 'suyo',
 'algo',
 'todos',
 'hube',
 'estaba',
 'son',
 'vuestra',
 'seamos',
 'el',
 'sea',
 'estariamos',
 'estare',
 'estuvieran',
 'tuvisteis',
 'estas',
 'a',
 'estabamos',
 'otras',
 'su',
 'por',
 'estuvisteis',
 'tambien',
 'fuisteis',
 'tuviesen',
 'entre',
 'estabais',
 'tendran',
 'eres',
 'estos',
 'tendreis',
 'vosotros',
 'hayas',
 'y',
 'tu',
 'vuestros',
 'habia',
 'estuviste',
 'algunos',
 'estad',
 'fue',
 'tenian',
 'tuvieseis',
 'estabas',
 'habias',
 'suya',
 'vosotras',
 'estuviera',
 'seremos',
 'muchos',
 'ya',
 'ha',
 'estas',
 'hubieses',
 'sentido',
 'tenidas',
 'te',
 'ellas',
 'estuvieron',
 'hubieras',
 'fueras',
 'los',
 'estudio',
 'Estudio',
 'estudios',
 'cuales']
def clean_text(text):
  		text = text.lower()
  		text = unidecode(text)
  		text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
  		words = text.split()
  		filtered_words = [word for word in words if word.lower() not in stopw]
  		return " ".join(filtered_words).strip()

#print(clean_text("hola como estás?"))

auth_config = weaviate.AuthApiKey(api_key=os.environ['WEAVIATE_API_KEY'])
	
client = weaviate.Client(url=os.environ['WEAVIATE_URL'], auth_client_secret=auth_config, additional_headers={
	        "X-OpenAI-Api-Key": os.environ['OPENAI_API_KEY'], # Replace with your OpenAI key
	        })

	

retriever = WeaviateHybridSearchRetriever(
    	client=client,
    	index_name="Evicardio",
    	text_key="content",
    	attributes=[],
    	create_schema_if_missing=True,
)

retriever.alpha = 0.25

retriever.k = 3

#lotr = MergerRetriever(retrievers=[retriever, retriever2])

class CustomRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager: None):
        # Use your existing retriever to get the documents
        print("RAW QUERY", query)

        # Process the input string
        for replacement_dict in replacement_list:
            for key, value in replacement_dict.items():
                query = query.replace(key, f"{key} {value}")

        query = clean_text(query)
        query = query.replace('latinoamerica', 'latinoamérica')
        query = query.replace('latino america', 'latinoamérica')
        print("CLEAN QUERY", query)
        
        documents = retriever.get_relevant_documents(query)
        return documents
custom_retriever = CustomRetriever()
	
	
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
	
llm_question = ChatOpenAI()
	
question_generator = LLMChain(llm=llm_question, prompt=CONDENSE_QUESTION_PROMPT)
	
llm = ChatOpenAI(temperature=0, verbose=True, max_tokens=500)
doc_chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
	
chain = ConversationalRetrievalChain(
	    retriever=custom_retriever,
	    question_generator=question_generator,
	    combine_docs_chain=doc_chain,
	    verbose=True, return_source_documents=True
	)
	
chain.combine_docs_chain.llm_chain.prompt = chat_prompt
a = chain({"question": "que son iecas", "chat_history": []})
print(a)
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
		#response = chain({"question": "que es el enalapril", "chat_history": []})
		#print(response)
	
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
			                #docs = lotr.get_relevant_documents(user_input)
			                res_get = requests.get(url).json()
			                res_get = res_get.get('credits')

			                if res_get == "OK":
			                    if len(st.session_state.ai) == 0:
			                        with get_openai_callback() as cb:
			                        	response = chain({"question": user_input, "chat_history": []})
			                        price = round(cb.total_cost,5)
			                        tokens = cb.total_tokens
			                        output = response['answer']
			                        docs = response['source_documents']
			                        raw_string = ''
			                        for d in range(len(docs)):
			                        	raw_string += f'Extracto {d+1}:\n'
			                        	raw_string += docs[d].page_content.replace('\n', ' ')
			                        	raw_string += '\n'
			                        	#raw_string += f"Página {str(docs[d].metadata['page'])}"
			                        	raw_string += '\n\n'
			                        print(len(docs))
			                        print("DOCS")
			                        st.session_state['data'].append(raw_string)
			                        st.session_state.ai.append(output)
			                        st.session_state.past.append(user_input)
			                        st.session_state['generated'].append(output)
			                        data = {
    			                        "question": user_input,
    			                        "answer": output,
    			                        "context": raw_string,
    			                        "tokens": tokens,
    			                        "price": price
			                        }
			                        json_data = json.dumps(data)

			                        requests.post(url, data=json_data)

			                    elif len(st.session_state.ai) == 1:
			                        chat_history = [(st.session_state['past'][-1], st.session_state['generated'][-1])]
			                        with get_openai_callback() as cb:
			                        	response = chain({"question": user_input, "chat_history": chat_history})
			                        price = round(cb.total_cost,5)
			                        tokens = cb.total_tokens
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
			                        data = {
    			                        "question": user_input,
    			                        "answer": output,
    			                        "context": raw_string,
    			                        "tokens": tokens,
    			                        "price": price
			                        }
			                        json_data = json.dumps(data)
			                        requests.post(url, data=json_data)
			                    else:
			                        chat_history = [(st.session_state['past'][-2], st.session_state['generated'][-2]), (st.session_state['past'][-1], st.session_state['generated'][-1])]                
			                        with get_openai_callback() as cb:
			                        	response = chain({"question": user_input, "chat_history": chat_history})
			                        price = round(cb.total_cost,5)
			                        tokens = cb.total_tokens
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
			                        data = {
    			                        "question": user_input,
    			                        "answer": output,
    			                        "context": raw_string,
    			                        "tokens": tokens,
    			                        "price": price
			                        }
			                        json_data = json.dumps(data)
			                        requests.post(url, data=json_data)
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
		stx.scrollableTextbox(patient_data,height = 500)
