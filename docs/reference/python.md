# Python API

The functions and classes most programs need are importable from `tinydistill` directly; the rest are in its
modules.

## Using a model

::: tinydistill.export.Predictor

::: tinydistill.export.Cascade

## Labelling

::: tinydistill.task.load_task

::: tinydistill.task.Task

::: tinydistill.data.read_examples

::: tinydistill.teacher.load_teacher

::: tinydistill.teacher.LabelScorer
    options:
      members: [score, score_many, fingerprint]

::: tinydistill.teacher.ScoreCache

## Training and export

::: tinydistill.student.train_student

::: tinydistill.student.distillation_loss

::: tinydistill.student.few_gold

::: tinydistill.student.few_shot_examples

::: tinydistill.export.export_onnx

::: tinydistill.export.export_explain

## Evaluation and charts

::: tinydistill.evaluate.accuracy

::: tinydistill.evaluate.macro_f1

::: tinydistill.evaluate.top_k_accuracy

::: tinydistill.plots
    options:
      members: [soft_labels, speed_accuracy, training, per_label, confusions, cascade, save_all]

## Deploying

::: tinydistill.web.build_site

::: tinydistill.hub.push
