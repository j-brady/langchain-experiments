# Chat with your documents

## Install

Make a virtual environment. Inside the repository top level directory you can use poetry or pip to install.

`poetry install`

or 

`pip install .`

You also need an OpenAI API KEY.

`export OPENAI_API_KEY=<insert_api_key>`

## Run

Assuming you have a directory with `.pdf` files you can simply provide a path to that directory.

`python app.py path/to/pdfs/`

You can then chat with your documents in the gradio app.

Chat history is saved in `path/to/pdfs/history.txt` by default but the location can be changed using the `--history path/to/pdfs/alternate_history.txt`.

Alternatively, you can run the `jupyter-notebook`.

```bash
jupyter-notebook ChatNotebookExample.ipynb
```


## Todo

- Chat history is stored on disk and used to repopulate the `ConversationBufferMemory` however, this is not updated on the gradio app which makes it hard to see the history unless you directly read the history file.

- Integrate into emacs or other text editor for more effective note taking.

