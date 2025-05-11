import json

with open("reference.jsonl") as f1, open("phi.jsonl") as f2, open("qwen.jsonl") as f3, open("cleaned.jsonl", "w") as fout:
    for line1, line2, line3 in zip(f1, f2, f3):
        reference = json.loads(line1)
        phi = json.loads(line2)
        qwen = json.loads(line3)
        if phi["output"] is not None and qwen["output"] is not None and reference["output"] is not None and phi["input"] == qwen["input"] and qwen["input"] == reference["input"]:
            out_dict = {"input": reference["input"], "reference": reference["output"],
                        "phi": phi["output"], "qwen": qwen["output"]}
            print(json.dumps(out_dict), file=fout)
