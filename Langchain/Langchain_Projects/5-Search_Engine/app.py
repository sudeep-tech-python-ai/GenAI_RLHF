import streamlit as st 
from langchain_groq import ChatGroq 
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType,AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
import os 
from dotenv import load_dotenv
load_dotenv()

## Arxiv and Wikipedia tools 
arxiv_api_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=arxiv_api_wrapper)
wiki_api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

search=DuckDuckGoSearchRun(name='Search')

st.title("Langchain - Chat with search")
"""
In this example, we're using `StreamlitCallbakHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more Langchain Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings 
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:",type='password')

if "messages" not in st.session_state:
    st.session_state['messages']=[
        {"role":"assistant","content":"Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:=st.chat_input(placeholder='What is machine learning?'):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)
    llm=ChatGroq(groq_api_key=api_key,model='llama3-8b-8192',streaming=True)
    tools=[search,arxiv,wiki]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

    with st.chat_message('assistant'):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant','content':response})
        st.write(response)
