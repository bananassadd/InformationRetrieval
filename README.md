# InformationRetrieval

ArXiv CS.CL Hybrid Search Engine

Follow these steps to run the search engine on your own computer:

1. Download the Repository
Clone this repository to your local machine or download it as a ZIP file and extract it.

"git clone https://github.com/your-username/InformationRetrieval.git"

"cd InformationRetrieval"


2. Install the Required Libraries
You need Python installed on your computer. Open your terminal or command prompt inside the project folder and run this command to install all the necessary dependencies:

"pip install streamlit pandas rank_bm25 sentence-transformers pyspellchecker arxiv PyMuPDF requests"


3. Run the Search Engine
Once everything is installed, start the Streamlit interface by running this exact command in your terminal:

"python -m streamlit run app.py"