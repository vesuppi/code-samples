from datasets import load_dataset

dataset = load_dataset("code_search_net", "python", split='train')
print(dataset[0])


