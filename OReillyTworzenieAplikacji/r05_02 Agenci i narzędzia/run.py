from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Pytanie "Jaki jest pierwiastek kwadratowy z populacji stolicy kraju, w którym w 2016 roku odbyły się igrzyska olimpijskie?"
question = """What is the square root of the population of the capital of the
Country where the Olympic Games were held in 2016?"""
agent.run(question)