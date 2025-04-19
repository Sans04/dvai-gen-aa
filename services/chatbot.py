from langchain_ollama import ChatOllama
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Annotated
from langgraph.graph import StateGraph
from agents.nejm_agents import NEJMGeneralAgent
import asyncio


class FastAPICallbackHandler(BaseCallbackHandler):
    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.queue = queue
        self.loop = loop

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        asyncio.run_coroutine_threadsafe(self.queue.put(token), self.loop)

    @property
    def ignore_chat_model(self) -> bool: return False
    @property
    def ignore_llm(self) -> bool: return False
    @property
    def ignore_chain(self) -> bool: return True
    @property
    def ignore_agent(self) -> bool: return True
    @property
    def raise_error(self) -> bool: return True

class ChatBot:
    def __init__(self):
        self.llm = ChatOllama(
            model="gemma3:4b",
            base_url="http://192.168.198.134:11434",
            streaming=True,
            callbacks=[]
        )
        self.vectorstore = self.load_vectorstore()
        self.graph = self.build_graph()

    def load_vectorstore(self):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.load_local(
            "archive/vector_index",
            embeddings,
            allow_dangerous_deserialization=True
        )


    def answer_question(self, question):
        output = self.graph.invoke({"question": question})
        return output
    
    def build_graph(self):
        state_type = Annotated[dict, lambda old, new: {**old, **new}]
        graph = StateGraph(state_type)

        def general_agent(state):
            question = state.get("question", "")
            agent = NEJMGeneralAgent(self, self.vectorstore)
            response = agent.execute(question)

            return {"response": response}
        
        graph.add_node("general_agent", general_agent)

        graph.set_entry_point("general_agent")
        graph.add_edge("general_agent", "__end__")

        return graph.compile()
