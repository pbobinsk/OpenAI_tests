from dotenv import load_dotenv

load_dotenv()
import openai
import pandas as pd

def chat_completion(prompt, model='gpt-4', temperature=0):
    res = openai.ChatCompletion.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=temperature
    )
    out = res['choices'][0]['message']['content']
    print(out)
    return out

l_sector = ['sklepy spożywcze', 'restauracje', 'fast foody', 'apteki', 'stacje paliw', 'sklepy z elektroniką']
l_city = ['Warszawa', 'Paryż', 'Berlin']
l_size = ['mały', 'średni', 'duży'] 

f_prompt = """ 
Rola: jesteś ekspertem w pisaniu treści, z dużym doświadczeniem w marketingu bezpośrednim. Twoje cechy to umiejętność dobrego pisania, kreatywność, stosowanie różnych brzmień i stylów, dogłębne zrozumienie potrzeb i preferencji odbiorców, niezbędne do przeprowadzania skutecznych bezpośrednich kampanii marketingowych.
Kontekst: będziesz pisać krótkie wiadomości e-mail, nie więcej niż dwa zdania, na potrzeby bezpośredniej kampanii marketingowej, reklamującej nową usługę płatności dla sklepów internetowych.
Sklep docelowy ma trzy następujące cechy:
- Branża działalności: {sektor}
- Miasto, w którym się znajduje sklep: {miasto}
- Wielkość sklepu: {wielkość}
Zadanie: napisz krótką wiadomość e-mail na potrzeby bezpośredniej kampanii marketingowej. Aby napisać wiadomość, wykorzystaj umiejętności zdefiniowane w roli. Ważne jest, aby wiadomość uwzględniała reklamowaną usługę i charakterystykę sklepu, dla którego jest przeznaczona. 
"""

f_sub_prompt = "{sektor}, {miasto}, {wielkość}"

df = pd.DataFrame()
for sector in l_sector:
 for city in l_city:
  for size in l_size:
   for i in range(3): ## Każda kombinacja 3x.
    prompt = f_prompt.format(sektor=sector, miasto=city, wielkość=size)
    sub_prompt = f_sub_prompt.format(sektor=sector, miasto=city, wielkość=size)

    response_txt = chat_completion(prompt, model='gpt-3.5-turbo', temperature=1)

    new_row = {
      'prompt':sub_prompt, 
      'completion':response_txt}
    new_row = pd.DataFrame([new_row])
    df = pd.concat([df, new_row], axis=0, ignore_index=True)

df.to_csv("uzupełnienia.csv", index=False)
