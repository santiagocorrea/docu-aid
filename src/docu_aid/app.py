#from llama_index.agent.openai import OpenAIAgent
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from utils import load_docs, get_agent_tools, get_object_index, set_up_env
import streamlit as st
import pdb
import os 

def validate_api_key(api_key: str) -> bool:
    """
    Validates the provided API key by making a simple API call to list available models.

    Parameters:
    -----------
    api_key : str
        The API key to be validated.

    Returns:
    --------
    bool
        Returns `True` if the API key is valid, otherwise returns `False`.
    """
    from openai import OpenAI
    try:
        # Initialize the OpenAI client with the provided API key
        client = OpenAI(api_key=api_key)

        # Attempt to list available models as a test to validate the API key
        client.models.list()

        # If the API call succeeds, return True indicating the key is valid
        return True
    except Exception as e:
        # If an error occurs, display an error message and return False
        st.error(f"API Key validation failed: {e}")
        return False


@st.cache_resource
def initialize_app(_api_key: str):
    """
    Initializes the application by setting up the environment, loading documents, 
    creating agents and query engines, and returning the object index.

    Parameters:
    -----------
    _api_key : str
        The API key for OpenAI services.

    Returns:
    --------
    obj_index
        The object index created from the tools, which is used to interact with the application.
    """
    
    # Set the OpenAI API key as an environment variable
    os.environ["OPENAI_API_KEY"] = _api_key
    
    # Set up the environment (additional setup tasks can be defined in set_up_env)
    set_up_env()

    # Define the path to the data directory containing the documentation
    data_path = "../../data/"

    # Load documents from the specified data path
    docs, docs_dic = load_docs(data_path)

    # Create agents, query engines, and tools from the loaded documents
    agents, query_engines, tools = get_agent_tools(docs, docs_dic)

    # Generate the object index from the created tools
    obj_index = get_object_index(tools)

    # Return the object index to be used in the application
    return obj_index
    

def main():
    """
    Main function to run the DocuAid application, an AWS SageMaker Documentation Helper.
    Handles API key validation, application initialization, and user interaction.
    """

    # Set the title of the Streamlit app
    st.title("DocuAid: An AWS SageMaker Documentation Helper")
    
    # Check if the OpenAI API key is stored in the session state
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ''

    # Input field for the API key (masked as a password)
    api_key_input = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        value=st.session_state.openai_api_key
    )

    # Validate the API key if provided
    if api_key_input:
        if validate_api_key(api_key_input):
            # Store the validated API key in session state
            st.session_state.openai_api_key = api_key_input
            st.success("API Key validated successfully!")
        else:
            st.stop()  # Stop execution if the API key is invalid

    # Initialize the app only if the API key is provided and valid
    if st.session_state.openai_api_key:
        obj_index = initialize_app(st.session_state.openai_api_key)

        # Store the initialization state in session state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.success("Application initialized successfully!")

        # Set up the main agent with tools and a system prompt
        top_agent = OpenAIAgent.from_tools(
            tool_retriever=obj_index.as_retriever(similarity_top_k=3),
            system_prompt="""\
                You are an agent designed to answer queries about a set of given AWS SageMaker documentation.
                Please always use the tools provided to answer a question. You can rely on previous knowledge if needed.\
                """,
            verbose=True,
        )   

        # Input field for the user's query about AWS SageMaker
        query = st.text_input("Enter your question about AWS SageMaker:")

        if query:
            # Get the response from the agent
            response = top_agent.chat(query)
            sources_filename = []
            sources_filepath = []

            # Extract source information from the response, if available
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    sources_filename.append(node.to_dict()['node']['metadata']['file_name'])
                    sources_filepath.append(node.to_dict()['node']['metadata']['file_path'])
            
            # Display the response
            st.header("Response:")
            st.write(str(response))

            # Display the sources used in generating the response
            st.header("Sources:")
            for file in set(sources_filename):
                st.write(f"- {file}")
    else:
        # Prompt the user to enter a valid API key if none is provided
        st.warning("Please enter a valid OpenAI API Key to proceed.")


if __name__ == "__main__":
    main()