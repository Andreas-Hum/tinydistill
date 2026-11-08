"""tinydistill: distil an LLM classification prompt into a small, fast model.

The command line runs the whole pipeline; the same steps are available from Python:

    from tinydistill import load_task, load_teacher, train_student, export_onnx, Predictor

or in a few lines, scikit-learn style: ``Distiller("task.yaml").fit(texts).predict(texts)``.
"""

from .data import Example, read_examples
from .distiller import Distiller
from .evaluate import accuracy, macro_f1
from .export import Predictor, export_onnx
from .student import distillation_loss, few_gold, few_shot_examples, train_student
from .task import Task, load_task
from .teacher import LabelScorer, ScoreCache, load_teacher

__all__ = [
    "Distiller",
    "Example",
    "LabelScorer",
    "Predictor",
    "ScoreCache",
    "Task",
    "accuracy",
    "distillation_loss",
    "export_onnx",
    "few_gold",
    "few_shot_examples",
    "load_task",
    "load_teacher",
    "macro_f1",
    "read_examples",
    "train_student",
]
__version__ = "0.6.0"
