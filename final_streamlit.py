import streamlit as st
from langchain.chains import ConversationalRetrievalChain
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_openai import ChatOpenAI
import streamlit.components.v1 as components

#from chromadb import Client  # Chroma client
import pandas as pd
#from langchain.schema import HumanMessage, AIMessage
# Set the page configuration
st.set_page_config(
    page_title="AUBFM-BOT",
    page_icon='💬',
    # layout='wide'
)


# Function to create a list of Document objects from the CSV file
def create_documents_from_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)  # Load the CSV as a pandas DataFrame
    documents = []

    # Iterate over each row in the CSV to create Document objects
    for idx, row in df.iterrows():
        question = row['Question']
        answer = row['Answer']
        page_content = f"Question: {question}\nAnswer: {answer}"
        metadata = {"source": uploaded_file.name, "row": idx + 1}
        doc = Document(metadata=metadata, page_content=page_content)
        documents.append(doc)

    return documents
# Function to process the CSV and generate embeddings without persisting Chroma
def load_documents_from_chroma(uploaded_file, api_key):
    documents = create_documents_from_csv(uploaded_file)  # Create documents
    embeddings = OpenAIEmbeddings(openai_api_key=api_key,model="text-embedding-3-small")  # Initialize OpenAI embeddings

    # Create Chroma vector store in memory without persisting
    chroma_vector = Chroma.from_documents(documents, embeddings)
    retriever = chroma_vector.as_retriever(search_type="mmr", k=3)
    return retriever

# Function to create a prompt based on query, history, and retrieved documents
def create_prompt(query, history, retrieved_docs):
    if len(history) > 3:
        history = history[-3:]  # Keep only the last 5 interactions
    # Create a structured prompt with the conversation history, current query, and retrieved context
    history_prompt = "\n".join([f"User: {msg[0]}\nBot: {msg[1]}" for msg in history])
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = (
f"You are an expert assistant at the Faculty of Medicine, AUB. Greet the user and introduce yourself appropriately.\n"
f"Here is the conversation history so far:\n"
f"{history_prompt}\n\n"
f"Here are some relevant documents: {context}\n\n"
f"And here is the user's latest question: {query}\n\n"
f"Answer the inquiry in a detailed, professional, and consistent tone, using the relevant documents and being aware of the conversation history. "
f"However, if none of the relevant documents can answer the question or provide any related information, and there's nothing in the history to answer it, respond with: "
f"'Sorry, I don't have an answer to your inquiry. Kindly check our website: https://www.aub.edu.lb/FM/Pages/default.aspx.'\n"
f"If the question is a follow-up asking for more details on a previously asked question, and no related information is found, use the conversation history to provide an answer."

    )
    return prompt

# Function to handle the query and interaction with the LLM
def query_llm(retriever, query,api_key):
    # Retrieve relevant documents using the updated method
    retrieved_docs = retriever.invoke(query)

    # Create the prompt including the retrieved context
    prompt = create_prompt(query, st.session_state.messages, retrieved_docs)

    # Use the prompt to query the LLM
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0),
        retriever=retriever,
        return_source_documents=True,

    )
    # result = qa_chain({'question': prompt, 'chat_history': []})
    result = qa_chain.invoke({'question': prompt, 'chat_history': []})

    response = result['answer']

    # Store the query and response in the session state
    st.session_state.messages.append((query, response))
    return response

# Sidebar input for OpenAI API key and file upload
def input_fields():
    with st.sidebar:
        openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    return openai_api_key, uploaded_file

# Processing the CSV and creating retriever from Chroma
def process_documents(api_key, uploaded_file):
    if uploaded_file and api_key:
        # Initialize retriever from local Chroma vector store
        retriever = load_documents_from_chroma(uploaded_file, api_key)
        return retriever
    return None

# Boot process to load the retriever and display the chat interface
def boot():
    api_key, uploaded_file = input_fields()
    retriever = process_documents(api_key, uploaded_file)

    if retriever:
        st.session_state.retriever = retriever
        st.success("Embeddings have been created and stored in-memory Chroma.")

        # Initialize chat history if it doesn't exist
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display past conversation
        for message in st.session_state.messages:
            st.chat_message('you').write(message[0])
            st.chat_message('AUBFM BOT').write(message[1])

        # Handle new input from the user
        if query := st.chat_input("Ask a question"):
            st.chat_message("you",avatar="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQPPQUMVStKOsQ3MVoy7SMgNsE8O-FqNETAJQ&s").write(query)
            response = query_llm(st.session_state.retriever, query,api_key)
            with st.chat_message('AUBFM BOT', avatar='https://alumni.aub.edu.lb/s/1716/images/gid2/editor/ProgressIndicator/FM_Emblem_638349910177900810.jpg'):
                st.write(response)
         #   st.chat_message("AUBFM BOT",avatar=st.image('C:\\Users\\Abir\\Documents\\capstone project\\Code\\chatbot_pycharm\\aubfm.jpeg')).write(response)

    else:
        st.warning("Please upload a CSV file and enter your OpenAI API key.")

# Main function to handle the page navigation
def main():

    # Add a sidebar to select the page
    # page = st.sidebar.selectbox("Select a page:", ["Introduction", "Chatbot"])
    # st.sidebar.title("Pages")
    st.sidebar.markdown(
        """
        <style>
        .sidebar-title {
            font-size: 20px;
            font-weight: bold;
            color: black;
        }
        .red-text {
            color: red;
        }
        </style>
        <div class="sidebar-title">Pages</div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio("** **",["Introduction", "AUB-FM chatbot"],
                            captions=[
                                "About AUB-FM.",
                                "QA bot",
                            ],
                            )


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



    elif page == "AUB-FM chatbot":
        st.header('AUB FM chatbot')
        #   st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/2_%E2%AD%90_context_aware_chatbot.py)')
        with st.expander("Disclaimer"):
            st.write("""
            While the chatbot aims to offer accurate and helpful responses, it might occasionally make **mistakes** or misinterpret your question. Please ensure you **double-check important details** and seek professional guidance when necessary.
            """)

        boot()

if __name__ == '__main__':
    main()
