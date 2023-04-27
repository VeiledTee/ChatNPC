import json

names = {
    "Evelyn Stone-Brown": "",
    "Caleb Brown": "",
    "Jack McCaster": "",
    "Peter Satoru": "",
    "Melinda Deek": "",
    "Sandra Ratengen": "",
}


# Define a function to convert a name to the desired format
def format_name(name):
    # Replace spaces with underscores
    formatted_name = name.replace(" ", "_")
    # Convert to lowercase
    formatted_name = formatted_name.lower()
    return formatted_name


# Loop over the names and format them
for key in names:
    formatted_name = format_name(key)
    names[key] = formatted_name

# Save the dictionary to a file
with open("Text Summaries/characters.json", "w") as f:
    json.dump(names, f)
