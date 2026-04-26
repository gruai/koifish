# https://github.com/modelscope/evalscope
# evalscope eval --model /home/cys/rnd/lic/Models/Qwen3-4B --datasets piqa --limit 10
# evalscope eval --model /home/cys/rnd/lic/Models/Qwen3-4B --datasets hellaswag --limit 1000

from evalscope import run_task, TaskConfig
import json

def demo():
    # Configuration
    TASK_CFG = {
        "task": "lm_eval",
        "model_args": {
            "model": "hf",
            "pretrained": "/home/cys/rnd/lic/Models/Qwen3-4B",  # or your local Qwen3 path
            "trust_remote_code": True,
            "device": "cuda",  # use "cpu" if no GPU
            "batch_size": 4,
            "dtype": "auto",
            # "load_in_4bit": True,  # uncomment to save GPU memory
        },
        "tasks": [
            "piqa",
            "arc_easy",
            "arc_challenge",
            "hellaswag"
        ],
        "limit": None,  # set to 10 for a quick test
        "output_path": "./eval_results_qwen3"
    }

    # Run evaluation
    results = run_task(task_cfg=TASK_CFG)

    # Show results
    print("\n=== Evaluation Results ===")
    for task, metrics in results.items():
        if task in TASK_CFG["tasks"]:
            acc = metrics.get("acc", 0.0)
            print(f"{task:<15} accuracy: {acc:.4f}")

    # Save full results
    with open("qwen3_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

def demo_1():
    task_cfg = TaskConfig(
        model='/home/cys/rnd/lic/Models/Qwen3-4B',
        datasets=['gsm8k', 'arc'],
        limit=50


        
    )
    run_task(task_cfg)

if __name__ == "__main__":
    demo_1()