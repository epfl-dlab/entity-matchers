import argparse
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace the word embedding path in all the RDGCN args")
    parser.add_argument("--new_path", required=True, type=str, help="New path for the word embeddings")
    args = parser.parse_args()
    for file in os.listdir("../experiments/args_best"):
        if "rdgcn" in file:
            path = f"../experiments/args_best/{file}"
            with open(path) as json_file:
                json_data = json.load(json_file)
            json_data["word_embed"] = args.new_path
            with open(path, 'w') as outfile:
                json.dump(json_data, outfile, indent=4)
    print("Successfully updated path!")
