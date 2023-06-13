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


See the example `jupyter-notebook` `ChatNotebookExample.ipynb`.

To run:

```bash
jupyter-notebook ChatNotebookExample.ipynb
```

```python
from pathlib import Path

from app import Chat

pdfs = Path("pdfs/")
chat = Chat(pdfs)
chat.start()
```

Chat history is saved in `path/to/pdfs/history.txt` by default but the location can be changed using the `previous_messages` keyword argument when instantiating the Chat class:

```python
pdfs = Path("pdfs/")
chat = Chat(pdfs,previous_messages="alternate_history.txt")
chat.start()
```



## Todo
- Optimize chunking of text
- Extend to other formats
- Integrate into emacs or other text editor for more effective note taking.

