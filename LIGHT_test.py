import pickle
import json

with open('Data/light_data.pkl', 'rb') as f:
    data = pickle.load(f)
print(data[1])
print(type(data))

with open('Data/light_environment.pkl', 'rb') as f:
    data = pickle.load(f)
# print(data)
with open("Data/light_environment.json", "w") as outfile:
    json.dump(data, outfile)
print(type(data))

with open('Data/light_unseen_data.pkl', 'rb') as f:
    data = pickle.load(f)
print(data[1])
print(type(data))
# from nomic.gpt4all import GPT4All
# m = GPT4All()
# m.open()
# m.prompt('write me a story about a lonely computer')
# # from pyllamacpp.model import Model
# #
# # def new_text_callback(text: str):
# #     print(text, end="", flush=True)
# #
# # model = Model(ggml_model='./models/gpt4all-model.bin', n_ctx=512)
# # generated_text = model.generate("Once upon a time, ", n_predict=55)
# # print(generated_text)