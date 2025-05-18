import ollama
import os


model = "gemma3:4b"
input_file = "data/grocery_list.txt"
output_file = "data/categorized_grocery_list.txt"

if not os.path.exists(input_file):
    print(f"Input file {input_file} does not exist")
    exit(1)

#read the uncaegorized grocery list
with open(input_file, "r") as f:
    items = f.read().strip()

prompt = f"""
You are an assistant that categorizes and sorts grocery items.
Here is a list of all the grocery items:
{items}

Please:

1. Categorize these items into appropriate categories like produce, bakery, dairy, Beverage, snacks etc.
2. Sort them alphabetically within each category.
3. Present the categorized list in clear and organized manner, using bullet points or numbering.
    
"""

# Send the prompt and get the response
try:
    response = ollama.generate(model=model, prompt=prompt)
    generated_text = response.get("response", "")
    print(f"===========Generated text===========: \n {generated_text}")

    # Write the categorized list to the output file
    with open(output_file, "w") as f:
        f.write(generated_text.strip())

    print(f"Categorized grocery list has been saved to '{output_file}'")

except Exception as e:
    print("An error occurred:", str(e))