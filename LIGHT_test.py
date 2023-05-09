import json
import os
import pickle
import time

import parlai.core.build_data as build_data
from parlai.core.build_data import DownloadableFile
from parlai.tasks.light_dialog.builder import build_from_db

RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/light/light-dialog-processed-small7.pkl',
        'light_data.pkl',
        '7c83cf49818586db9999ea67a4a6ad087afbd91c26ed629a9f00e21d0b84058f',
        zipped=False,
    ),
    DownloadableFile(
        'http://parl.ai/downloads/light/light-unseen-processed2.pkl',
        'light_unseen_data.pkl',
        '489b98d08dd94eaf1ba95439d04200ccc54623ade056839f87a5c4207bc5699c',
        zipped=False,
    ),
    DownloadableFile(
        'http://parl.ai/downloads/light/light-environment.pkl',
        'light_environment.pkl',
        '162389202f22063e1c32af7f9261aac13d20fc05598388d1e9748735996ec016',
        zipped=False,
    ),
    DownloadableFile(
        'http://parl.ai/downloads/light_project/wild_chats/contents.txt',
        'current_chats.txt',
        'c708fe62692f239a2b35025d71722c7607f863ffa110aa118f2e1d0fa7db4730',
        zipped=False,
    ),
    DownloadableFile(
        'http://parl.ai/downloads/genderation_bias/genderation_bias.tgz',
        'genderation_bias.tgz',
        '9a0252c6bb778757ac60dee9df23a169192f4a853ceb2b530af2343abeb1498a',
    )
]


def download_LIGHT(dpath: str = 'Data/LIGHT'):
    for downloadable_file in RESOURCES:
        downloadable_file.download_file(dpath)

# with open("Data/light_data.pkl", "rb") as f:
#     data = pickle.load(f)
# print(data[1])
# print(type(data))

# with open("Data/light_environment.pkl", "rb") as f:
#     data = pickle.load(f)
# # print(data)
# # with open("Data/light_environment.json", "w") as outfile:
# #     json.dump(data, outfile)
# # print(data.keys())
# # print(data['categories'])
# # # print(data['rooms'])
# # # print(data['neighbors'])
# # print(data['characters'].keys())
# invalid_ids = []
# for i in range(1, 1785):
#     try:
#         # x = data['characters'][i]['corrected_name']
#         print(i, data['characters'][i]['corrected_name'])
#     except KeyError:
#         invalid_ids.append(i)
# # print(type(data))
# # print(invalid_ids)
# print(data['characters'][39]['corrected_name'])
# print(f"\t{data['characters'][39]['personas']}")
# print(f"\t{data['characters'][39]['desc']}")
# print(data['characters'][67]['corrected_name'])
# print(f"\t{data['characters'][67]['personas']}")
# print(f"\t{data['characters'][67]['desc']}")
# print(data['characters'][168]['corrected_name'])
# print(f"\t{data['characters'][168]['personas']}")
# print(f"\t{data['characters'][168]['desc']}")
# print(data['characters'][776]['corrected_name'])
# print(f"\t{data['characters'][776]['personas']}")
# print(f"\t{data['characters'][776]['desc']}")
# print(data['characters'][840]['corrected_name'])
# print(f"\t{data['characters'][840]['personas']}")
# print(f"\t{data['characters'][840]['desc']}")
# print(data['characters'][1508]['corrected_name'])
# print(f"\t{data['characters'][1508]['personas']}")
# print(f"\t{data['characters'][1508]['desc']}")

# with open("Data/light_environment.pkl", "rb") as f:
#     data = pickle.load(f)
# print(data)
# print(data[1])
# print(type(data))
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

"""
Fisherman indexes:
39, 67, 168, 776, 840, 1508
    Get info:    
    print(data['characters'][39]['corrected_name'])
    print(f"\t{data['characters'][39]['personas']}")
    print(f"\t{data['characters'][39]['desc']}")
    print(data['characters'][67]['corrected_name'])
    print(f"\t{data['characters'][67]['personas']}")
    print(f"\t{data['characters'][67]['desc']}")
    print(data['characters'][168]['corrected_name'])
    print(f"\t{data['characters'][168]['personas']}")
    print(f"\t{data['characters'][168]['desc']}")
    print(data['characters'][776]['corrected_name'])
    print(f"\t{data['characters'][776]['personas']}")
    print(f"\t{data['characters'][776]['desc']}")
    print(data['characters'][840]['corrected_name'])
    print(f"\t{data['characters'][840]['personas']}")
    print(f"\t{data['characters'][840]['desc']}")
    print(data['characters'][1508]['corrected_name'])
    print(f"\t{data['characters'][1508]['personas']}")
    print(f"\t{data['characters'][1508]['desc']}")

"""
if __name__ == '__main__':
    directory = 'Data/LIGHT'
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            with open(f"{directory}/{filename}", "rb") as f:
                data = pickle.load(f)
            if type(data) == dict:
                print(filename, data.keys())
                print(data['categories'].keys())
                print(data['rooms'].keys())
                print(data['neighbors'].keys())
                print(data['characters'].keys())
                print(data['objects'].keys())
                for k in ['categories', 'rooms', 'neighbors', 'characters', 'objects']:
                    for i in range(min(data[k].keys()), max(data[k].keys()) + 1):
                        try:
                            print(i)
                            print(data[k][i])
                            # print(data[k][i])
                        except KeyError:
                            pass
            else:
                pass
                # descriptions: dict = {}
                # agents: dict = {}
                # dupe_items: int = 0
                # total_items: int = 0
                # dupe_characters: int = 0
                # total_characters: int = 0
                # print(filename)
                # for i in range(len(data)):
                #     # print(data[0].keys())
                #     # print(data[0]["all_descriptions"])
                #     # print(data[0]["all_descriptions"].keys())
                #     for item, desc in data[i]["all_descriptions"].items():
                #         # print(item, "|", desc)
                #         if item not in descriptions.keys():
                #             descriptions[item] = desc
                #         elif descriptions[item] != desc:
                #             print(f"Item: {item}\n\tSaved: {descriptions[item]}\n\tNew: {desc}")
                #             dupe_items += 1
                #         total_items += 1
                #     for agent in data[i]['agents']:
                #         # print(agent['name'], "|", agent['persona'])
                #         if agent['name'] not in agents.keys():
                #             agents[agent['name']] = agent['persona']
                #         elif agents[agent['name']] != agent['persona']:
                #             print(f"Character: {agent['name']}\n\tSaved: {agents[agent['name']]}\n\tNew: {agent['persona']}")
                #             dupe_characters += 1
                #         total_characters += 1
                # # print(descriptions)
                # # print(agents)
                # print(f"{filename} done")
                # print(f"Item dupes: {dupe_items}")
                # print(f"Item total: {total_items}")
                # print(f"Character dupes: {dupe_characters}")
                # print(f"Character total: {total_characters}")
