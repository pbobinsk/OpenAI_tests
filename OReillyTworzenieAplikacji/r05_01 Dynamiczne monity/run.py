from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 

template = """Pytanie: {question}
Rozumuj krok po kroku.
Odpowiedź: """
prompt = PromptTemplate(template=template, input_variables=["question"])
llm = ChatOpenAI(model_name="gpt-4")
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = """Jaka jest populacja stolicy kraju, w którym
w 2016 roku odbyły się igrzyska olimpijskie?
"""

response = llm_chain.run(question)
print(response) 
