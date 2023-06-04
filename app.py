import time
import json
from pathlib import Path
from typing import Optional
from enum import Enum

import typer
import gradio as gr
from IPython.display import display, Markdown
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationTokenBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict

cli = typer.Typer()


class FtypeEnum(Enum):
    pdf = "pdf"
    txt = "txt"


class ChainTypeEnum(Enum):
    stuff = "stuff"
    refine = "refine"
    map_reduce = "map_reduce"


class MemoryTypeEnum(Enum):
    token = ConversationTokenBufferMemory
    standard = ConversationBufferMemory


class LlmEnum(Enum):
    gpt3 = "gpt-3.5-turbo"


def load_data(directory, ftype: FtypeEnum):
    directory = Path(directory)
    match ftype:
        case ftype.pdf:
            glob = directory / "*.pdf"
        case ftype.txt:
            glob = directory / "*.txt"
        case _:
            raise "NotImplemented"
    loader = DirectoryLoader("", glob=str(glob))
    docs = loader.load_and_split()
    return docs


def load_previous_messages(path):
    if Path(path).is_file():
        with open(path) as messages:
            previous_messages = json.load(messages)
            # return messages_from_dict(previous_messages)
            return previous_messages
    else:
        return []


def save_messages(memory, path):
    previous_messages = load_previous_messages(path)
    current_messages = messages_to_dict(memory.chat_memory.messages)
    previous_messages.extend(current_messages[len(previous_messages) :])
    with open(path, "w") as history:
        json.dump(previous_messages, history)


template = """
**Question:** %s

**Answer:** %s

<hr style="height:2px;border-width:0;color:black;background-color:black">

"""

hr = """

<hr style="height:2px;border-width:0;color:black;background-color:black">

"""


def convert_type(dic):
    if dic.get("type") == "ai":
        return "Answer"
    elif dic.get("type") == "human":
        return "Question"
    else:
        return dic.get("type")


class Chat:
    def __init__(
        self,
        qa,
        previous_messages: Path,
        memory,
        auto_save: bool = True,
        reload_previous_messages=True,
        response_template=template,
    ):
        self._qa = qa
        self.previous_messages = previous_messages
        self._memory = memory
        self.auto_save = auto_save
        self.response_template = response_template
        if reload_previous_messages:
            self.load_chat_memory()

    def respond(self, query):
        result = self._qa({"question": query})
        answer = self.response_template % (query, result["answer"])
        if self.auto_save:
            save_messages(self._memory, self.previous_messages)
        return display(Markdown(answer))

    def load_chat_memory(self):
        self.memory.chat_memory.messages = messages_from_dict(
            load_previous_messages(self.previous_messages)
        )

    def start(self):
        n = 0
        while True:
            if n == 0:
                # display previous messages at beginning of session
                previous = load_previous_messages(self.previous_messages)
                for i in previous:
                    message_type = convert_type(i)
                    message_string = f"**{message_type}:** {i['data']['content']}"
                    if message_type == "Answer":
                        message_string += hr
                    display(Markdown(message_string))

            time.sleep(0.5)
            query = input("You: ")

            if query.lower() == "bye":
                break
            else:
                n += 1
                self.respond(query)

    @property
    def memory(self):
        return self._memory

    @property
    def qa(self):
        return self._qa


@cli.command()
def main(
    directory: Path,
    history: Optional[Path] = None,
    ftype: FtypeEnum = FtypeEnum.pdf,
    gradio: bool = False,
    chain_type: ChainTypeEnum = ChainTypeEnum.refine,
    memory_type: MemoryTypeEnum = MemoryTypeEnum.standard,
    max_token_limit: int = 4097,
    temperature: float = 0.2,
    llm: LlmEnum = LlmEnum.gpt3,
):
    docs = load_data(directory, ftype)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        # metadatas=[{"source": str(i)} for i in range(len(docs))]
    )

    llm = ChatOpenAI(model_name=llm.value, temperature=temperature)

    match memory_type:
        case MemoryTypeEnum.standard:
            memory = ConversationBufferMemory(
                memory_key="chat_history", output_key="answer", return_messages=True
            )
        case MemoryTypeEnum.token:
            memory = ConversationTokenBufferMemory(
                llm=llm,
                memory_key="chat_history",
                output_key="answer",
                return_messages=True,
                max_token_limit=max_token_limit,
            )

    if history == None:
        history_path = directory / "history.txt"
    else:
        history_path = Path(history)
    memory.chat_memory.messages = messages_from_dict(
        load_previous_messages(history_path)
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type=chain_type.value,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    if gradio:
        with gr.Blocks() as demo:
            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            clear = gr.Button("Clear")

            def respond(query, chat_history):
                result = qa({"question": query})
                save_messages(memory, history_path)
                chat_history.append((query, result["answer"]))
                time.sleep(1)
                return "", chat_history

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)
        demo.launch()
    return qa, memory


if __name__ == "__main__":
    cli()
