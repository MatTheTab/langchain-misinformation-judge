import streamlit as st
from langchain import HuggingFaceHub
from langchain.chat_models import ChatHuggingFace
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import WikipediaQueryRun
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun

CHUNKS_LIMIT = 5
load_dotenv()
st.sidebar.title("Misinformation Detection from YouTube Videos")
st.sidebar.write("Enter a YouTube URL to detect misinformation.")

def load_video(url):
    try:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
        transcript = loader.load()
        return transcript
    except Exception as e:
        st.error(f"Error loading video: {e}")
        return None

def process_video(transcript):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(transcript)
    llm = OpenAI(temperature=0.7)
    wiki_api_wrapper = WikipediaAPIWrapper()
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)
    tools = [
        Tool(name="Wikipedia", func=wiki_tool.run, description="Use Wikipedia to check if the information in the statement is True, respond only with True or False")
    ]

    correction_tools = [
        Tool(name="Wikipedia", func=wiki_tool.run, description="Use Wikipedia to correct misinformation in the statement")
    ]

    agent = initialize_agent(
        tools=tools,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        llm=llm,
        handle_parsing_errors=True
    )

    correction_agent = initialize_agent(
        tools=correction_tools,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        llm=llm,
        handle_parsing_errors=True
    )

    results = []
    for i, chunk in enumerate(chunks):
        if i > CHUNKS_LIMIT:
            st.write(f"### REACHED CHUNK LIMIT")
            break
        claim = chunk.page_content
        st.write(f"### Chunk {i + 1}")
        st.write(f"**Original Text:** {claim}")

        response = agent.run(claim)
        st.write(f"**Model Response:** {response}")

        if "not true" in response.lower() or "false" in response.lower():
            try:
                correction = correction_agent.run(claim)
                st.write(f"**Corrected Information:** {correction}")
            except Exception as e:
                st.write("**Misinformation Detected But Not Enough Tokens To Correct**")
                st.error(e)
        else:
            st.write("**No misinformation detected**")

def main():
    st.title("Misinformation Detection from YouTube Videos")
    youtube_url = st.text_input("Enter YouTube Video URL:")

    if st.button("Analyze Video"):
        if youtube_url:
            st.write("Loading video and analyzing...")
            transcript = load_video(youtube_url)
            
            if transcript:
                st.write("Video loaded successfully. Analyzing transcript...")
                process_video(transcript)
        else:
            st.warning("Please enter a valid YouTube URL.")

if __name__ == "__main__":
    main()
