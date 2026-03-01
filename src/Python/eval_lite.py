import os
os.environ["PYDEVD_DISABLE_SYS_LOG"] = "1"
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
import torch
# PyTorch < 2.1.0: No official Python 3.12 support; PyTorch 2.1.0+: Full Python 3.12 support; PyTorch 2.4.0+: Optimized for Python 3.12
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")  # Should return True if GPU is set up

from transformers import AutoModelForCausalLM
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

# opencompass --models /home/cys/rnd/lic/Models/Qwen3-0.6B --datasets /home/cys/Datasets/hellaswag
MODEL_NAME = "/home/cys/rnd/lic/Models/Qwen3-0.6B"
BENCHMARKS = "/home/cys/Datasets/hellaswag"

evaluation_tracker = EvaluationTracker(output_dir="./results")
pipeline_params = PipelineParameters(
    launcher_type=ParallelismManager.NONE,
    max_samples=2
)

model = AutoModelForCausalLM.from_pretrained(
  MODEL_NAME, device_map="auto"
)
config = TransformersModelConfig(model_name=MODEL_NAME, batch_size=1)
model = TransformersModel.from_model(model, config)

pipeline = Pipeline(
    model=model,
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    tasks=BENCHMARKS,
)

results = pipeline.evaluate()
pipeline.show_results()
results = pipeline.get_results()