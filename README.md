This is an AI project.
It is a custom research bot.
We need to pass the location of a pdf file to it.
It will then parse the file and embed it into a vector store.
When we ask a query to it, the query will also be embedded and similar results will be retrieved.
Then using llama2 (locally), chat completion is performed.

To use it, first create a virtual environment and activate it.
Once activated, pip install requirements.txt.
Then copy the codes:embed_doc.py and query_bot.py

Note:-Since we are using llama2, we will need to install git, llama2 model(huggingface), and visual C++ Build tools.
The bot performs with moderate accuracy.
It has been tested on 3 pdfs.


Note:- you can use other model (multi-qa-MiniLM-L6-cos-v1)
