import time
import json
from pathlib import Path
from enum import Enum
from collections import OrderedDict

import pandas as pd
import numpy as np
from IPython.display import display, Markdown
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationTokenBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict

# cli = typer.Typer()


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



<details> 
<summary><b>Sources (click to expand):</b><summary>


%s


</details>
<hr style="height:2px;border-width:0;color:black;background-color:black">


"""

hr = """

<hr style="height:2px;border-width:0;color:black;background-color:black">

"""

source_template = """


<details> <summary>   %d) <i>%s</i> [%d]</summary>

    %s

</details>

    
"""


def format_source_details(source_documents) -> str:
    source_details = ""
    for num, source_document in enumerate(source_documents):
        page_content = source_document.page_content
        metadata = source_document.metadata
        source = metadata.get("source", "")
        index = metadata.get("index", 0)
        source_details += source_template % (num + 1, source, index, page_content)
    return source_details


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
        directory: Path,
        ftype: FtypeEnum = FtypeEnum.pdf,
        memory_type: MemoryTypeEnum = MemoryTypeEnum.token,
        max_token_limit: int = 4097,
        chain_type: ChainTypeEnum = ChainTypeEnum.refine,
        auto_save: bool = True,
        previous_messages: Path = Path("history.txt"),
        reload_previous_messages=True,
        response_template: str = template,
        llm: LlmEnum = LlmEnum.gpt3,
        temperature: float = 0.2,
    ):
        self.directory = Path(directory)
        self.previous_messages = self.directory / previous_messages
        self.auto_save = auto_save
        self.response_template = response_template

        self._docs = load_data(self.directory, ftype)
        self.update_metadata()
        embeddings = OpenAIEmbeddings()
        self._vectorstore = Chroma.from_documents(
            self._docs,
            embeddings,
            # metadatas=[
            #     {"source": self._docs[i].metadata["source"]+f"/{i}"}
            #     for i in range(len(self._docs))
            # ],
        )

        llm = ChatOpenAI(model_name=llm.value, temperature=temperature)
        match memory_type:
            case MemoryTypeEnum.standard:
                self._memory = ConversationBufferMemory(
                    memory_key="chat_history", output_key="answer", return_messages=True
                )
            case MemoryTypeEnum.token:
                self._memory = ConversationTokenBufferMemory(
                    llm=llm,
                    memory_key="chat_history",
                    output_key="answer",
                    return_messages=True,
                    max_token_limit=max_token_limit,
                )
            case _:
                raise TypeError("I don't know this memory type")

        if reload_previous_messages:
            self.load_chat_memory()

        self._qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type=chain_type.value,
            retriever=self._vectorstore.as_retriever(),
            memory=self._memory,
            return_source_documents=True,
        )

    def update_metadata(self):
        doc_dic = OrderedDict(page_content=[], source=[])
        for i in self._docs:
            doc_dic["page_content"].append(i.page_content)
            doc_dic["source"].append(i.metadata["source"])
        df = pd.DataFrame(doc_dic)
        df["source_index"] = 0
        for name, group in df.groupby("source"):
            df.loc[df.source == name, "source_index"] = np.arange(len(group))
        for ind, row in df.iterrows():
            self._docs[ind].metadata["index"] = row.source_index
        self.df = df

    def respond(self, query):
        result = self._qa({"question": query})
        sources = format_source_details(result["source_documents"])
        answer = self.response_template % (query, result["answer"], sources)
        if self.auto_save:
            save_messages(self._memory, self.previous_messages)
        return display(Markdown(answer))

    def load_chat_memory(self):
        self._memory.chat_memory.messages = messages_from_dict(
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

    @property
    def vectorstore(self):
        return self._vectorstore

    @property
    def docs(self):
        return self._docs


# @cli.command()
# def main(
#    directory: Path,
#    history: Optional[Path] = None,
#    ftype: FtypeEnum = FtypeEnum.pdf,
#    # gradio: bool = False,
#    chain_type: ChainTypeEnum = ChainTypeEnum.refine,
#    memory_type: MemoryTypeEnum = MemoryTypeEnum.standard,
#    max_token_limit: int = 4097,
#    temperature: float = 0.2,
#    llm: LlmEnum = LlmEnum.gpt3,
# ):

# if gradio:
#     with gr.Blocks() as demo:
#         chatbot = gr.Chatbot()
#         msg = gr.Textbox()
#         clear = gr.Button("Clear")

#         def respond(query, chat_history):
#             result = qa({"question": query})
#             save_messages(memory, history_path)
#             chat_history.append((query, result["answer"]))
#             time.sleep(1)
#             return "", chat_history

#         msg.submit(respond, [msg, chatbot], [msg, chatbot])
#         clear.click(lambda: None, None, chatbot, queue=False)
#     demo.launch()
# return qa, memory


# if __name__ == "__main__":
#     cli()
