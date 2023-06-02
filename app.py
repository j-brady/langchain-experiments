import time
import json
from pathlib import Path
from typing import Optional
from enum import Enum

import typer
import gradio as gr
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict

cli = typer.Typer()


class FtypeEnum(Enum):
    pdf = "pdf"
    txt = "txt"


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
    if Path(path).exists():
        with open(path) as messages:
            previous_messages = json.load(messages)
            # return messages_from_dict(previous_messages)
            return previous_messages
    else:
        return []


def save_messages(memory, path):
    previous_messages = load_previous_messages(path)
    current_messages = messages_to_dict(memory.chat_memory.messages)
    previous_messages.extend(current_messages)
    with open(path, "w") as history:
        json.dump(previous_messages, history)


@cli.command()
def main(directory: Path, history: Optional[Path] = None, ftype: FtypeEnum = FtypeEnum.pdf, gradio: bool=False):
    docs = load_data(directory, ftype)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        docs, embeddings, metadatas=[{"source": str(i)} for i in range(len(docs))]
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
    memory = ConversationBufferMemory(
        memory_key="chat_history", output_key="answer", return_messages=True
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
        chain_type="stuff",
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
