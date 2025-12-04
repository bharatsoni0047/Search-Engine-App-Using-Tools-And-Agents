# import required libraries
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.callbacks import StreamlitCallbackHandler
from langgraph.prebuilt import create_react_agent
from langchain_classic import hub   # ye sahi hai ab


# page config
st.set_page_config(
    page_title="Web Search Bot",
    page_icon="search",
    layout="centered"
)

# title
st.title("search Chat with Real-time Web Search")

# description
st.markdown("""
In this example, we're using `StreamlitCallbackHandler` to show live agent thinking.  
More examples → [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent)
""")


# sidebar - groq api key
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Groq API Key", type="password", help="Get free key at console.groq.com")

if not api_key:
    st.info("please add your groq api key to continue")
    st.stop()


# chat history setup
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "hi! i'm a chatbot that can search the web in real-time. ask me anything!"}
    ]


# show previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# user input
if prompt := st.chat_input("what do you want to know?"):
    
    # add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # llm setup
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-70b-versatile",   # faster & better than old llama3-8b
        temperature=0.6,
        streaming=True
    )

    # tools
    tools = [DuckDuckGoSearchRun()]

    # react prompt from hub
    react_prompt = hub.pull("hwchase17/react")

    # create react agent
    agent_executor = create_react_agent(llm, tools, react_prompt)

    # run agent with live thinking
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        response = agent_executor.invoke(
            {"messages": st.session_state.messages},
            {"callbacks": [st_cb]}
        )

        answer = response["messages"][-1].content
        
        # save & show reply
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.write(answer)


# footer
st.markdown("---")
st.caption("built with ❤️ langchain v1 • groq • langgraph • streamlit | dec 2025 working version")