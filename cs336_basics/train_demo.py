from tokenization import train_bpe
import cProfile
import pstats
import pickle
# 1. Create the Profiler
profiler = cProfile.Profile()

path = r"C:\Users\Arpit Rawat\Desktop\cs336\assignment1-basics\cs336_basics\data\owt_train.txt"
#path = (pathlib.Path(__file__).resolve().parent.parent) / "tests" / "fixtures" / "corpus.en"
#special_token = bytes(" ","utf-8")
#dataset = load_dataset("roneneldan/TinyStories")
#train_data = dataset['train']
#val_data = dataset['validation']

#print(f"Total training stories: {len(train_data)}")
#print(f"Total testing stories: {len(val_data)}")

if __name__ == "__main__":

    profiler.enable()
    vocab, merges = train_bpe(input_path=path, vocab_size=32000, special_tokens=["<|endoftext|>"])

    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(20)

    state = {
        "vocab": vocab,
        "merges": merges
    }
    path = r"assignment1-basics\cs336_basics\owt_train.pkl"
    with open(path, "wb") as f:
        pickle.dump(state, f)