import os
import json

def compile_dataset():
    # We will grab all JSONL files from all three environments
    data_dirs = {
        "c": "./c-data", 
        "cpp": "./cpp-data", 
        "python": "./py-data"
    }
    
    master_data = []

    for lang, dir_path in data_dirs.items():
        if not os.path.exists(dir_path): 
            continue
        
        for filename in os.listdir(dir_path):
            if not filename.endswith(".jsonl"): 
                continue
            
            # Extract battery mode from filename (e.g., battery-saver_benchmark_results.jsonl)
            mode = filename.replace("_benchmark_results.jsonl", "")
            
            with open(os.path.join(dir_path, filename), "r") as f:
                for line in f:
                    if not line.strip(): 
                        continue
                    record = json.loads(line.strip())
                    
                    # Tag each record with the language and battery mode
                    record["language"] = lang
                    record["battery_mode"] = mode
                    master_data.append(record)
                    
    # We will output this directly into the website's public folder later
    os.makedirs("./website/public/data", exist_ok=True)
    out_path = "./website/public/data/master_dataset.json"
    
    with open(out_path, "w") as f:
        json.dump(master_data, f, indent=2)
        
    print(f"✅ Compiled {len(master_data)} benchmark records into {out_path}")

if __name__ == "__main__":
    compile_dataset()
