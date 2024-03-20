# Large Language Models (LLMs)
We are living in an LLM-led AI world. LLMs are colossal AI systems, trained on massive datasets of text and code, and possess remarkable abilities, including generating human-quality text, translating languages, writing different kinds of creative content, and even answering our questions in an informative way. 

In this project, the following LLM tasks are performed: 

## 1. Mistral 7B model
It is a quantized version of Mistral AI large model. It is small enough to run locally. [Code here](https://github.com/sumitdeole/llm_from_rag_to_chatbots/blob/main/code/1_llm_primer.ipynb)

## 2. Create a chatbot
In this notebook, I will be a chatbot with the capability to retain information from previous prompts and responses, enabling it to maintain context throughout the conversation. [Code here](https://github.com/sumitdeole/llm_from_rag_to_chatbots/blob/main/code/2_simple-chatbot.ipynb)

## 3. Retrieval Augmented Generation (RAG) model
RAG combines information retrieval with language models and improves the ability of LLM tremendously. It first searches for relevant facts in external sources, and then feeds those facts to the language model alongside the user's prompt. This helps the model generate more accurate and factual responses, even on topics beyond its initial training data. In this notebook, I will go beyond pre-trained models to customizing LLMs. [Code here](https://github.com/sumitdeole/llm_from_rag_to_chatbots/blob/main/code/3_rag.ipynb)

## 4. RAG with memory
After exploring LLMs, chatbots, and RAG, I now try to put them all together to create a powerful tool: a RAG chain with memory. To this end, I will use the `ConversationalRetrievalChain`, a LangChain chain for RAG with memory. [Code here](https://github.com/sumitdeole/llm_from_rag_to_chatbots/blob/main/code/4_rag_chatbot.ipynb)


I'll install two libraries: Langchain and Llama.cpp. 

**LangChain** is a framework that simplifies the development of applications powered by large language models (LLMs)

**llama.cpp** enables us to execute quantized versions of models.

In addition to notebook testing, all these models are deployed using Streamlit and tested locally.
