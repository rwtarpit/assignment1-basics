import pickle

path = r"assignment1-basics\cs336_basics\demo_tokenizer.pkl"
with open(path,"rb") as f:
    loaded_dict = pickle.load(f)
    
print(loaded_dict["vocab"])