import time
import json
import re
from pathlib import Path
from enum import Enum
from collections import OrderedDict
from typing import List, Optional

import pandas as pd
import numpy as np
from IPython.display import display, Markdown
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, SeleniumURLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryBufferMemory,
)
from langchain.schema import messages_from_dict, Document


class FtypeEnum(Enum):
    pdf = "pdf"
    txt = "txt"
    url = "url"


class ChainTypeEnum(Enum):
    stuff = "stuff"
    refine = "refine"
    map_reduce = "map_reduce"


class MemoryTypeEnum(Enum):
    token = ConversationTokenBufferMemory
    standard = ConversationBufferMemory
    summary_buffer = ConversationSummaryBufferMemory


class LlmEnum(Enum):
    gpt3 = "gpt-3.5-turbo"
    gpt4 = "gpt-4"
    gpt3_16k = "gpt-3.5-turbo-16k"


class SearchTypeEnum(Enum):
    maximum_marginal_relevance = "mmr"
    similarity = "similarity"


def clean_text_and_split(documents):
    new_documents = []
    for document in documents:
        document.page_content = re.sub(
            " +", " ", document.page_content.replace("\n", "")
        )
        new_documents.append(document)
    splitter = RecursiveCharacterTextSplitter()
    documents = splitter.split_documents(new_documents)
    return documents


def load_urls(path):
    with open(path) as f:
        urls = f.readlines()
        loader = SeleniumURLLoader(urls=urls, browser="firefox")
        docs = clean_text_and_split(loader.load())
    return docs


def load_data(
    directory: Path, ftype: FtypeEnum, glob: Optional[str] = None
) -> List[Document]:
    directory = Path(directory)
    if directory.is_dir():
        pass
    else:
        raise FileNotFoundError(f"{directory} does not exist!")

    if glob is not None:
        try:
            files = directory.glob(glob)
            next(files)
        except StopIteration:
            raise ValueError("No files found matching pattern")

    else:
        match ftype:
            case ftype.pdf:
                glob = "**/*.pdf"
            case ftype.txt:
                glob = "**/*.txt"
            case ftype.url:
                docs = load_urls(directory / "urls.txt")
                return docs
            case _:
                raise ValueError("File type not implemented")
    loader = DirectoryLoader(
        directory, glob=str(glob), show_progress=True, use_multithreading=True
    )
    docs = clean_text_and_split(loader.load())
    return docs


def load_previous_messages(path: Path) -> List:
    if Path(path).is_file():
        with open(path) as messages:
            previous_messages = json.load(messages)
            # return messages_from_dict(previous_messages)
            return previous_messages
    else:
        return []


def save_chat_history(result: dict, source_path: Path):
    source_documents = [i.dict() for i in result.get("source_documents", [])]
    answer = result["answer"]
    # chat_history = result["chat_history"].dict()
    question = result["question"]
    source_path = Path(source_path)
    if source_path.is_file():
        with open(source_path) as f:
            sources = json.load(f)
    else:
        sources = []
    source_dict = dict(
        source_documents=source_documents, answer=answer, question=question
    )
    sources.append(source_dict)
    with open(source_path, "w") as f:
        json.dump(sources, f)


def load_chat_history(source_path: Path) -> List:
    source_path = Path(source_path)
    if source_path.is_file():
        with open(source_path) as f:
            sources = json.load(f)
            history = []
            for i in sources:
                history.extend(
                    [
                        {
                            "type": "human",
                            "data": {
                                "content": i["question"],
                                "additional_kwargs": {},
                                "example": False,
                            },
                        },
                        {
                            "type": "ai",
                            "data": {
                                "content": i["answer"],
                                "additional_kwargs": {
                                    "source_documents": [
                                        Document(**j)
                                        for j in i.get("source_documents", [])
                                    ]
                                },
                                "example": False,
                            },
                        },
                    ]
                )
            return history
    else:
        return []


template = """
**Question:** %s

**Answer:** %s


<details> 

<summary><b>Sources (click to expand):</b></summary>


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


def format_source_details(source_documents: List[Document]) -> str:
    source_details = ""
    for num, source_document in enumerate(source_documents, start=1):
        page_content = source_document.page_content
        metadata = source_document.metadata
        source = metadata.get("source", "")
        index = metadata.get("index", 0)
        source_details += source_template % (num, source, index, page_content)
    return source_details


def convert_type(dic: dict) -> str:
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
        glob: Optional[str] = None,
        memory_type: MemoryTypeEnum = MemoryTypeEnum.summary_buffer,
        max_token_limit: int = 1300,
        chain_type: ChainTypeEnum = ChainTypeEnum.refine,
        auto_save: bool = True,
        previous_messages: Path = Path(".history.json"),
        reload_previous_messages=True,
        response_template: str = template,
        llm: LlmEnum = LlmEnum.gpt3,
        temperature: float = 0.2,
        k: int = 3,
        persist_directory: Optional[str] = None,
        search_type: SearchTypeEnum = SearchTypeEnum.similarity,
    ):
        self.directory = Path(directory)
        self.previous_messages = self.directory / previous_messages
        self.auto_save = auto_save
        self.response_template = response_template
        self.k = k
        self.glob = glob
        self.search_type = search_type.value
        embeddings = OpenAIEmbeddings()
        if persist_directory is None:
            self._docs = load_data(self.directory, ftype, self.glob)
            self.update_metadata()
            self._vectorstore = Chroma.from_documents(
                self._docs, embeddings, persist_directory=persist_directory
            )
        elif Path(persist_directory).is_dir():
            raise NotImplementedError("Persistence not implemented yet")
            # try:
            # self._vectorstore = Chroma.from_documents(
            #     embeddings, persist_directory=persistence_directory
            # )
            # print(f"Loaded persistent vectorstore from {persistence_directory}")
            # # except:
            #     # self._vectorstore = Chroma.from_documents(
            #     #     self._docs, embeddings, persist_directory=persistence_directory
            #     # )
            #     # self._vectorstore.persist()

        else:
            raise NotADirectoryError(f"{persist_directory} is not a directory")

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
            case MemoryTypeEnum.summary_buffer:
                self._memory = ConversationSummaryBufferMemory(
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
            retriever=self._vectorstore.as_retriever(
                search_type=self.search_type, search_kwargs={"k": self.k}
            ),
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
        source_documents = result["source_documents"]
        if self.auto_save:
            save_chat_history(result, source_path=self.previous_messages)
            # save_messages(self._memory, self.previous_messages)
        return result, source_documents

    def format_response(self, query, result, source_documents):
        sources = format_source_details(source_documents)
        answer = self.response_template % (query, result, sources)
        return display(Markdown(answer))

    def load_chat_memory(self):
        self._memory.chat_memory.messages = messages_from_dict(
            load_chat_history(self.previous_messages)
        )

    def start(self):
        n = 0
        while True:
            if n == 0:
                # display previous messages at beginning of session
                previous = load_chat_history(self.previous_messages)
                previous = messages_from_dict(previous)
                # previous = load_chat_history(self.previous_messages)
                for i in range(0, len(previous), 2):
                    self.format_response(
                        query=previous[i].content,
                        result=previous[i + 1].content,
                        source_documents=previous[i + 1].additional_kwargs.get(
                            "source_documents", []
                        ),
                    )

            time.sleep(0.5)
            query = input("You: ")

            if query.lower() == "bye":
                break
            else:
                n += 1
                result, source_documents = self.respond(query)
                self.format_response(query, result["answer"], source_documents)

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
