import ollama

#chat example
response = ollama.chat(
    model="gemma3:4b",
    messages=[{"role":"user", "content":"How do horses sleep?"}],
    stream=True, #With stream we can see chunks of info come in ass they are generated.
)
for chunk in response:
    print(chunk["message"]["content"], end="", flush=True)

#generate example
res = ollama.generate(
    model="gemma3:4b",
    prompt="why is the sky blue? Give me a short one sentence answer"
)

#show
print(ollama.show("gemma3:4b"))

#Creating a new Model with the modelfile

system= "You are a quick and witty agent, who is sassy, to the point and succinct with your replies."
ollama.create(model="sassy",from_="gemma3:4b", system=system)

result= ollama.generate(
    model="sassy",
    prompt="How do horses sleep?",
)

print(result["response"])

#delete models
ollama.delete("sassy")
