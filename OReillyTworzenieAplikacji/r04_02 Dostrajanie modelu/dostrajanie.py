import openai

ft_file = openai.File.create(
    file=open("uzupełnienia_prepared.jsonl", "rb"), purpose="fine-tune"
)
openai.FineTune.create(
    training_file=ft_file["id"], model="davinci", suffix="marketing_bezpośredni"
)