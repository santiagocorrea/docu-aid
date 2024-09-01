import glob
import os
from typing import List

from llama_index.agent.openai import OpenAIAgent
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SummaryIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    load_index_from_storage
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.objects import ObjectIndex
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.schema import IndexNode
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


def set_up_env() -> None:
    """
    Set up the environment by configuring global settings for the language model and embedding model.

    This function initializes the OpenAI language model with specific parameters and sets up
    a HuggingFace embedding model for use in the application.

    Returns:
        None
    """
    # Configure the language model (LLM) settings
    Settings.llm = OpenAI(
        temperature=0,  # Set to 0 for more deterministic outputs
        model="gpt-3.5-turbo"  # Specify the GPT-3.5 Turbo model
    )

    # Configure the embedding model settings
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="dunzhang/stella_en_400M_v5",
        trust_remote_code=True  # Allow execution of remote code (use with caution)
    )


def get_object_index(tools: List[object]) -> ObjectIndex:
    """
    Create and return an ObjectIndex from a list of tools.

    This function takes a list of tool objects and creates a VectorStoreIndex,
    which can be used for efficient similarity search and retrieval.

    Args:
        tools (List[object]): A list of tool objects to be indexed.

    Returns:
        ObjectIndex: An ObjectIndex instance containing the indexed tools.
    """
    # Create an ObjectIndex using VectorStoreIndex for efficient similarity search
    obj_index = ObjectIndex.from_objects(
        tools,
        index_cls=VectorStoreIndex,
    )
    
    return obj_index


def get_agent_tools(docs: list, docs_dic: dict) -> tuple:
    """
    Generates tools and agents for navigating and querying a list of documentation files.
    
    Parameters:
    -----------
    docs : list
        A list of document paths.
    docs_dic : dict
        A dictionary containing document content keyed by document titles.
    
    Returns:
    --------
    tuple
        A tuple containing:
        - agents: A dictionary where keys are document titles and values are agent objects.
        - query_engines: A dictionary where keys are document titles and values are query engine objects.
        - all_tools: A list of query engine tools created for each document.
    """

    # Initialize a sentence splitter for node parsing
    node_parser = SentenceSplitter()

    # Initialize dictionaries and lists for agents, query engines, nodes, and tools
    agents = {}
    query_engines = {}
    all_nodes = []
    all_tools = []

    # Iterate over each document in the provided list
    for idx, doc in enumerate(docs):
        # Extract and format the document title from its path
        doc_title = doc.split('/')[-1][:min(len(doc.split('/')[-1])-1, 50)].strip('.md')
        print(len(doc_title))

        # Parse document into nodes using the sentence splitter
        nodes = node_parser.get_nodes_from_documents(docs_dic[doc_title])
        all_nodes.extend(nodes)

        # Check if a vector index exists; if not, create and persist it
        if not os.path.exists(f"../../vector_index/{doc_title}"):
            print("Creating vector index for:", doc_title)
            vector_index = VectorStoreIndex(nodes)
            vector_index.storage_context.persist(
                persist_dir=f"../../vector_index/{doc_title}"
            )
        else:
            print("Loading vector index for:", doc_title)
            vector_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=f"../../vector_index/{doc_title}"),
            )

        # Build summary index from nodes
        summary_index = SummaryIndex(nodes)

        # Define query engines for both vector and summary indices
        vector_query_engine = vector_index.as_query_engine(llm=Settings.llm)
        summary_query_engine = summary_index.as_query_engine(llm=Settings.llm)

        # Define query engine tools for the document
        query_engine_tools = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name="vector_tool",
                    description=(
                        "Useful for questions related to specific aspects of"
                        f" {doc_title} (e.g., definitions, configuration,"
                        " requirements, or more)."
                    ),
                ),
            ),
            QueryEngineTool(
                query_engine=summary_query_engine,
                metadata=ToolMetadata(
                    name="summary_tool",
                    description=(
                        "Useful for any requests that require a holistic summary"
                        f" of EVERYTHING about {doc_title}. For questions about"
                        " more specific sections, please use the vector_tool."
                    ),
                ),
            ),
        ]

        # Initialize the agent with the provided tools and system prompt
        function_llm = OpenAI(model="gpt-4")
        agent = OpenAIAgent.from_tools(
            query_engine_tools,
            llm=function_llm,
            verbose=True,
            system_prompt=f"""\
                You are a specialized agent designed to answer queries about {doc_title}.
                You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
                """,
        )

        # Store the agent and query engine in their respective dictionaries
        agents[doc_title] = agent
        query_engines[doc_title] = vector_index.as_query_engine(
            similarity_top_k=2
        )

        # Create a tool for the document summary and add it to the list of all tools
        doc_summary = (
            f"This content contains AWS SageMaker documentation about {doc_title}. Use"
            f" this tool if you want to answer any questions about {doc_title}.\n"
        )
        doc_tool = QueryEngineTool(
            query_engine=agent,
            metadata=ToolMetadata(
                name=f"tool_{doc_title}",
                description=doc_summary,
            ),
        )
        all_tools.append(doc_tool)
    
    # Return the generated agents, query engines, and tools
    return agents, query_engines, all_tools
    

def load_docs(data_path: str) -> tuple:
    """
    Loads Markdown documents from the specified directory and returns them in a dictionary.

    Parameters:
    -----------
    data_path : str
        The path to the directory containing Markdown (.md) files.

    Returns:
    --------
    tuple
        A tuple containing:
        - docs: A list of file paths for all Markdown documents found in the directory.
        - docs_dic: A dictionary where keys are document titles and values are the content of the documents.
    """

    # Initialize an empty dictionary to store document content
    docs_dic = {}

    # Use glob to find all Markdown files in the specified directory
    docs = glob.glob(data_path + "*.md")

    # Iterate over each document found
    for doc in docs:
        # Extract and format the document title from its path
        doc_title = doc.split('/')[-1][:min(len(doc.split('/')[-1])-1, 50)].strip('.md')
        
        # Load the document content using SimpleDirectoryReader and store it in the dictionary
        docs_dic[doc_title] = SimpleDirectoryReader(
            input_files=[doc]
        ).load_data()
    
    # Return the list of document paths and the dictionary containing their content
    return docs, docs_dic


