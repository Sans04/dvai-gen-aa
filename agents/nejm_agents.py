from langchain_core.prompts import PromptTemplate

class NEJMGeneralAgent:
    def __init__(self, chat, vectorstore):
        self.chat = chat
        self.vectorstore = vectorstore
        self.prompt_template = """
        Frage:
        {question}

        Antwort:
        """
        self.prompt = PromptTemplate.from_template(self.prompt_template)

    def execute(self, question):
        docs = self.vectorstore.similarity_search(question, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        input = self.prompt.format(context=context, question=question)
        print(input)
        response = self.chat.llm.invoke(input)
        return response
