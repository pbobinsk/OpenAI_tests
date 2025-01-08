from langchain.llms import OpenAI
from langchain.chains import ConversationChain

chatbot_llm = OpenAI(model_name='text-ada-001')
chatbot = ConversationChain(llm=chatbot_llm , verbose=True)

response = chatbot.predict(input='Hello')
print(response)

response = chatbot.predict(input='Can I ask you a question? Are you an AI?')
print(response)