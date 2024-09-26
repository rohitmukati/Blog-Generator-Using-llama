import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers


## Function to get responce from my lamma model
def getLLamaresponse(input_text,no_words,blog_style):
    ## Lamma2 model
    llm = CTransformers(model="lamma\llama-2-7b-chat.ggmlv3.q8_0.bin",
                        model_type = "llama",
                        config={"max_new_tokens":256,
                                "temperature":0.01})
    
    ## Prompt Template
    template="""
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
            """
    
    prompt=PromptTemplate(input_variables=["blog_style","input_text",'no_words'],
                          template=template)
    
    ## Generate the ressponse from the LLama 2 model
    response=llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    print(response)
    return response


#Streamlit app ui
st.set_page_config(page_title="Genrate Blogs",
                   page_icon="H",
                   layout="centered",
                   initial_sidebar_state="collapsed")

st.header("Generate Blogs")
input_text = st.text_input("Enter the blog topic")

## Creating two more colum for additional two fields
col1,col2 = st.columns([5,5])
with col1:
    no_words = st.text_input("No of words")
with col2:
    blog_style = st.selectbox("Writing the blog for", 
                              ("Researcher","Data Scientist","Ai Engineer","Commom people"),index=0)
    
submit = st.button("Generate")
if submit:
    st.write(getLLamaresponse(input_text,no_words,blog_style))











