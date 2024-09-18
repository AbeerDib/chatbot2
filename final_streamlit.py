import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import streamlit.components.v1 as components


# Set the page configuration as the first Streamlit command

st.set_page_config(
    page_title="FM bot",
    page_icon='💬',
    # layout='wide'
)
# Paths
LOCAL_VECTOR_STORE_DIR = "C:\\Users\\Abir\\Documents\\capstone project\\Data\\final_chroma"
def load_documents_from_chroma(api_key):
    # Initialize Chroma vector store
    vectordb = Chroma(
        persist_directory=LOCAL_VECTOR_STORE_DIR,
        embedding_function=OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-small")
    )
    retriever = vectordb.as_retriever(search_type="mmr", k=4)
    return retriever

def create_prompt(query, history, retrieved_docs):
    if len(history) > 3:
        history = history[-3:]  # Keep only the last 3 interactions
    history_prompt = "\n".join([f"User: {msg[0]}\nBot: {msg[1]}" for msg in history])
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = (
        f"You are an expert assistant at the Faculty of Medicine, AUB. Answer to greetings..\n"
        f"Here is the conversation so far:\n"
        f"{history_prompt}\n\n"
        f"Here are some relevant documents: {context}\n\n"
        f"and here's the user's latest question: {query}\n\n"
        f"Answer to the inquiry in a detailed professional and consistent tone using the relevant documents "
        f"while being aware of the history. However, if none and only none of the relevant documents clearly answer/related to the question, or there's no information from the history to answer it respond with 'Sorry, I don't have an answer to your inquiry. Kindly check our website: https://www.aub.edu.lb/FM/Pages/default.aspx'."
    )
    return prompt

def query_llm(retriever, query, api_key):
    retrieved_docs = retriever.invoke(query)
    prompt = create_prompt(query, st.session_state.messages, retrieved_docs)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4", api_key=api_key, temperature=0),
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': prompt, 'chat_history': []})
    response = result['answer']

    st.session_state.messages.append((query, response))
    return response

def input_fields():
    st.sidebar.header("Configuration")
    st.session_state.openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

def process_documents(api_key):
    st.session_state.retriever = load_documents_from_chroma(api_key)

def boot():
    input_fields()

    if "openai_api_key" in st.session_state and st.session_state.openai_api_key:
        api_key = st.session_state.openai_api_key
        process_documents(api_key)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            st.chat_message('you').write(message[0])
            st.chat_message('AUBFM BOT').write(message[1])

        if query := st.chat_input():
            st.chat_message("you").write(query)
            response = query_llm(st.session_state.retriever, query, api_key)
            st.chat_message("AUBFM BOT").write(response)
    else:
        st.warning("Please enter your OpenAI API Key in the sidebar.")

def main():
    page = st.sidebar.selectbox("Select a page:", ["Introduction", "Chatbot"])

    if page == "Introduction":
        st.image("https://www.aub.edu.lb/fm/PublishingImages/AUB_Logo_FM_Horizontal_RGB.png", caption="",width=350)
        # Render the HTML in Streamlit

        # st.image("https://www.aub.edu.lb/fm/PublishingImages/AUB_Logo_FM_Horizontal_RGB.png", caption="",width=350)
        components.html(
            """
            <!DOCTYPE html>
            <html>
            <head>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
            * {box-sizing: border-box;}
            body {font-family: Verdana, sans-serif; margin: 0; padding: 0;}
            .mySlides {display: none;}
            img {vertical-align: middle; width: 100%; height: auto;}
    
            /* Slideshow container */
            .slideshow-container {
              max-width: 100%;
              position: relative;
              margin: auto;
              overflow: hidden;
            }
    
            /* Caption text */
            .text {
              color: #f2f2f2;
              font-size: 15px;
              padding: 8px 12px;
              position: absolute;
              bottom: 8px;
              width: 100%;
              text-align: center;
            }
    
            /* Number text (1/3 etc) */
            .numbertext {
              color: #f2f2f2;
              font-size: 12px;
              padding: 8px 12px;
              position: absolute;
              top: 0;
            }
    
            /* Fading animation */
            .fade {
              animation-name: fade;
              animation-duration: 1.5s;
            }
    
            @keyframes fade {
              from {opacity: .4} 
              to {opacity: 1}
            }
    
            /* On smaller screens, decrease text size */
            @media only screen and (max-width: 300px) {
              .text {font-size: 11px}
            }
            </style>
            </head>
            <body>
    
            <div class="slideshow-container">
    
            <div class="mySlides fade">
              <div class="numbertext">1 / 3</div>
              <img src="https://www.aub.edu.lb/fm/SliderResearch/Slide6.jpg" style="width:100%">
            </div>
    
            <div class="mySlides fade">
              <div class="numbertext">2 / 3</div>
              <img src="https://www.aub.edu.lb/fm/MD-Program/slider/IMG_10522.jpg" style="width:100%">
            </div>
    
            <div class="mySlides fade">
              <div class="numbertext">3 / 3</div>
              <img src="https://www.aub.edu.lb/fm/MSAO/slider/students.jpg" style="width:100%">
    
            </div>
    
            </div>
    
            <script>
            let slideIndex = 0;
            showSlides();
    
            function showSlides() {
              let i;
              let slides = document.getElementsByClassName("mySlides");
              for (i = 0; i < slides.length; i++) {
                slides[i].style.display = "none";  
              }
              slideIndex++;
              if (slideIndex > slides.length) {slideIndex = 1}    
              slides[slideIndex-1].style.display = "block";  
              setTimeout(showSlides, 5000); // Change image every 5 seconds
            }
            </script>
    
            </body>
            </html> 
            """,
            height=250,
        )

        st.write("")
        st.write("""\n
        ### Welcome to the AUB Faculty of Medicine Chatbot!
        
        This chatbot is here to help answer your questions related to the **Faculty of Medicine at the American University of Beirut (AUB)**. Whether you're a **current or prospective student** or a **faculty member**, you can ask about a wide range of topics related to the faculty.
        
        #### How to Use the Chatbot:
        1. **Be Specific**: For the best results, please be as clear and detailed as possible with your questions.
           
        2. **Faculty-Specific**: This chatbot is tailored to assist with queries about the Faculty of Medicine **only**. It is not able to provide information about other faculties or departments at AUB.
        
        3. **Potential Errors**: While the chatbot aims to offer accurate and helpful responses, it might occasionally make **mistakes** or misinterpret your question. Please ensure you **double-check important details** and seek professional guidance when necessary.
        
        By using this chatbot, you acknowledge that it serves as a helpful resource but is not a replacement for official AUB channels or professional advice.
        
        For more information about the Faculty of Medicine, please check [this link](https://www.aub.edu.lb/FM/Pages/default.aspx).
        """)

        # VIDEO_URL = "https://www.youtube.com/watch?v=bslp1ReeFsc"
        # st.video(VIDEO_URL)



    elif page == "Chatbot":
        st.header('AUB FM chatbot')
        #   st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/2_%E2%AD%90_context_aware_chatbot.py)')
        with st.expander("Disclaimer"):
            st.write("""
                While the chatbot aims to offer accurate and helpful responses, it might occasionally make **mistakes** or misinterpret your question. Please ensure you **double-check important details** and seek professional guidance when necessary.
                """)
        boot()

if __name__ == '__main__':
    main()
