"""A scikit-learn style estimator over the pipeline: ``Distiller(task).fit(texts).predict(texts)``."""

from __future__ import annotations

import inspect
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np

from .student import DEFAULT_STUDENT
from .task import Task, load_task


class NotFittedError(RuntimeError):
    """A Distiller was asked to predict before ``fit`` or ``load``."""


class Distiller:
    """Distil a task's prompt into a small ONNX classifier, scikit-learn style.

    ``fit`` scores every text with the teacher LLM (the soft labels), trains the student on them and exports
    it to ONNX; ``predict``, ``predict_proba`` and ``describe`` then run the exported model through a
    :class:`~tinydistill.export.Predictor`. It is a thin wrapper over
    :func:`~tinydistill.teacher.load_teacher`, :func:`~tinydistill.student.train_student` and
    :func:`~tinydistill.export.export_onnx`, with the same defaults as the command line::

        d = Distiller("task.yaml").fit(texts)
        d.predict(["How do I top up my card?"])
        d.save("model")  # a folder Predictor and Distiller.load read

    Labels are the task's label ids (``classes_``), in the task's order. ``get_params`` and ``set_params``
    return and set the constructor's arguments, so tools such as scikit-learn's ``clone`` work without
    tinydistill depending on scikit-learn.

    :param task: the task: a task file's path or a :class:`~tinydistill.task.Task`.
    :param teacher: the teacher LLM (a Hugging Face model name).
    :param student: the student encoder (a Hugging Face model name).
    :param epochs: passes over the texts.
    :param lr: the student's learning rate.
    :param batch_size: texts per training step.
    :param device: ``auto`` (a GPU when there is one), ``cpu`` or ``cuda``; for the teacher and the student.
    :param load_4bit: load the teacher in 4 bits (bitsandbytes, CUDA only).
    :param cache_dir: a folder for the teacher's scores. The cache files are the ones ``tinydistill label
        --cache-dir`` uses, so command-line runs and Distillers share them, and when every text is cached the
        teacher is not loaded at all.
    :param int8: also write an int8 dynamically quantised model, and predict with it.
    :param seed: the training seed.
    :param val_share: hold out this share of the texts and keep the epoch with the lowest loss on them (see
        :func:`~tinydistill.student.train_student`).
    :param ce_weight: weight of the cross-entropy term on the gold labels given to ``fit``.
    :param few_shot: show the teacher K of the gold labels given to ``fit`` per class as examples in its
        prompt (needs ``{examples}`` in the task's prompt), picked as ``train --gold-per-class`` picks them.
    :param few_shot_similar: show each text the M gold examples most similar to it (``{examples}`` before
        ``{text}``), from those K per class with ``few_shot``, else from every gold-labelled text. By default,
        when ``fit`` gets gold labels and the prompt has ``{examples}``, 16 of every gold-labelled text, as
        ``tinydistill label`` does with ``--gold-per-class``; 0 for none.
    :param probe: when ``fit`` gets gold labels, fit a linear probe on the teacher's hidden states with them
        (see :func:`~tinydistill.teacher.fit_probe`). The states are cached with the scores.
    :param probe_only: the probe alone is the teacher: the teacher scores no label and only gives the probe
        its hidden states, so labelling is much faster. By default (None), as ``tinydistill label`` does:
        the probe alone when ``fit`` gets gold labels; ``False`` also scores every label and averages the
        probe's probabilities with the teacher's (slower, a little more accurate). ``True`` needs gold labels
        in ``fit``.
    :param log: called with each progress message.
    """

    def __init__(
        self,
        task: str | Path | Task,
        teacher: str = "Qwen/Qwen2.5-1.5B-Instruct",
        student: str = DEFAULT_STUDENT,
        epochs: int = 10,
        lr: float = 1e-4,
        batch_size: int = 32,
        device: str = "auto",
        load_4bit: bool = False,
        cache_dir: str | Path | None = None,
        int8: bool = True,
        seed: int = 0,
        val_share: float = 0.0,
        ce_weight: float = 1.0,
        few_shot: int | None = None,
        few_shot_similar: int | None = None,
        probe: bool = True,
        probe_only: bool | None = None,
        log: Callable[[str], None] = print,
    ):
        self.task = task
        self.teacher = teacher
        self.student = student
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.load_4bit = load_4bit
        self.cache_dir = cache_dir
        self.int8 = int8
        self.seed = seed
        self.val_share = val_share
        self.ce_weight = ce_weight
        self.few_shot = few_shot
        self.few_shot_similar = few_shot_similar
        self.probe = probe
        self.probe_only = probe_only
        self.log = log

    def get_params(self, deep: bool = True) -> dict:
        """The constructor's arguments, by name."""
        return {name: getattr(self, name) for name in inspect.signature(type(self)).parameters}

    def set_params(self, **params) -> Distiller:
        """Set constructor arguments by name; returns the Distiller. Takes effect at the next ``fit``."""
        known = self.get_params()
        for name, value in params.items():
            if name not in known:
                raise ValueError(f"unknown parameter {name!r}; expected one of {', '.join(known)}")
            setattr(self, name, value)
        return self

    def _task(self) -> Task:
        return self.task if isinstance(self.task, Task) else load_task(self.task)

    def _gold(self, task: Task, gold: list) -> list:
        """Label ids, names or indices (multi-label: lists of them) as indices, None for no label."""
        from .data import _gold_index, _gold_set

        if task.multi_label:
            return [_gold_set(g, task, None) for g in gold]
        return [_gold_index(g, task, None) for g in gold]

    def _examples(self, task: Task, texts: list[str], gold: list | None) -> tuple[list | None, int]:
        """The teacher's prompt examples (the pool to pick from, with ``similar``) and how many similar ones
        each text shows: as ``tinydistill label`` with ``--few-shot`` and ``--few-shot-similar``, and by
        default 16 similar ones of every gold-labelled text when there are gold labels and room for them."""
        from .student import few_shot_examples

        k, m = self.few_shot, self.few_shot_similar
        prompt = task.prompt
        fits = "{examples}" in prompt and prompt.index("{examples}") < prompt.index("{text}")
        labelled = gold is not None and any(g is not None for g in gold) and not task.multi_label
        if k is None and m is None:  # the default
            k, m = 0, (16 if labelled and fits else 0)
        k, m = k or 0, m or 0
        if not (k or m):
            return None, 0
        if not labelled:
            raise ValueError("few_shot and few_shot_similar need gold labels in fit (single-label tasks)")
        if "{examples}" not in prompt or (m and not fits):
            raise ValueError(
                "examples need {examples} in the task's prompt, before {text} for few_shot_similar"
            )
        gold = cast(list, gold)
        if k:
            return few_shot_examples(texts, gold, k, self.seed), m
        return list(dict.fromkeys((t, g) for t, g in zip(texts, gold, strict=True) if g is not None)), m

    def _teacher_scores(
        self,
        task: Task,
        texts: list[str],
        states: bool,
        examples: list | None = None,
        similar: int = 0,
        only: bool = False,
    ) -> tuple[list, list | None]:
        """The teacher's label scores for ``texts`` and, with ``states``, each text's hidden state (float16,
        as cached) for the probe; with ``only`` the states alone (the scores are not used)."""
        import gc

        import torch

        from .teacher import LabelScorer, ScoreCache, iter_unique, load_teacher, quick_key

        only = states and only
        keys_file, quick = None, None
        if self.cache_dir is not None:  # as `tinydistill label --cache-dir` names and finds the files
            cache_dir = Path(self.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            keys_file = cache_dir / "teacher-keys.json"
            keys = json.loads(keys_file.read_text(encoding="utf-8")) if keys_file.exists() else {}
            quick = quick_key(
                self.teacher, task, self.device, self.load_4bit, examples, similar=similar, states_only=only
            )
            if quick in keys:
                cache = ScoreCache(cache_dir / f"teacher-{keys[quick]}.jsonl", states)
                if all(cache.has(t) for t in texts):
                    self.log(f"{self.teacher}: every text is in the cache, so the teacher is not loaded")
                    return [cache.get(t) for t in texts], [cache.state(t) for t in texts] if states else None
        self.log(f"loading teacher {self.teacher}")
        # a local teacher (the Distiller loads no other), which can keep its states
        scorer = cast(
            LabelScorer,
            load_teacher(self.teacher, task, self.device, self.load_4bit, examples=examples, similar=similar),
        )
        scorer.keep_states, scorer.states_only = states, only
        try:
            if keys_file is None:
                scores = scorer.score_many(texts)
                kept = [scorer.states[t].numpy().astype(np.float16) for t in texts] if states else None
                return scores, kept
            fingerprint = scorer.fingerprint()
            keys = json.loads(keys_file.read_text(encoding="utf-8")) if keys_file.exists() else {}
            keys[quick] = fingerprint
            tmp = keys_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(keys, indent=1), encoding="utf-8")
            tmp.replace(keys_file)
            cache = ScoreCache(keys_file.parent / f"teacher-{fingerprint}.jsonl", states)
            n = cache.fill(scorer, iter_unique(texts))
            self.log(f"{n} texts scored now, the rest from the cache")
            return [cache.get(t) for t in texts], [cache.state(t) for t in texts] if states else None
        finally:
            del scorer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def fit(self, texts: list[str], gold: list | None = None) -> Distiller:
        """Label ``texts`` with the teacher, train the student on its soft labels and export it.

        :param texts: the training texts.
        :param gold: optional gold labels, one per text (a label id, name or index, a list of them for a
            multi-label task, or None for a text without one), used in a cross-entropy term weighted by
            ``ce_weight`` as with ``tinydistill train --gold-file``, and (``probe``) to fit the probe.
        :returns: the Distiller, fitted.
        """
        import torch

        from .export import Predictor, export_onnx
        from .student import train_student
        from .teacher import combine_scores, fit_probe, pick_device, probe_scores

        texts = list(texts)
        task = self._task()
        if gold is not None:
            gold = self._gold(task, list(gold))
            if task.multi_label:  # an empty list of labels counts as no gold labels, as in --gold-file
                gold = [g or None for g in gold]
            if len(gold) != len(texts):
                raise ValueError(f"{len(gold)} gold labels for {len(texts)} texts")
        labelled = [i for i, g in enumerate(gold or []) if g not in (None, [])]
        if self.probe_only and not labelled:
            raise ValueError("probe_only needs gold labels in fit: the probe is fit on them")
        use_probe = bool(labelled) and (self.probe or bool(self.probe_only))
        only = use_probe and self.probe_only is not False  # the default with gold labels: the probe alone
        examples, similar = self._examples(task, texts, gold)
        if examples:
            self.log(
                f"each text's prompt: its {similar} most similar of {len(examples)} gold examples"
                if similar
                else f"few-shot prompt with {len(examples)} gold examples"
            )
        scores, states = self._teacher_scores(task, texts, use_probe, examples, similar, only)
        if use_probe:  # as `tinydistill label`: fit on the gold texts' states, then a member of the teacher
            multi = task.multi_label
            x = torch.from_numpy(np.stack(cast(list, states)).astype(np.float32))
            weight, bias = fit_probe(
                x[labelled], [cast(list, gold)[i] for i in labelled], len(task.labels), multi
            )
            probed = probe_scores(weight, bias, x, multi)
            scores = (
                probed
                if only
                else [combine_scores([s, p], multi) for s, p in zip(scores, probed, strict=True)]
            )
            self.log(f"probe fit on {len(labelled)} gold texts" + (", the teacher" if only else ""))
        s = torch.tensor(scores, dtype=torch.float64)
        probs = (s.sigmoid() if task.multi_label else s.softmax(-1)).float()  # the soft targets
        work = tempfile.TemporaryDirectory(prefix="tinydistill-")
        root = Path(work.name)
        train_student(
            texts,
            probs,
            task.labels,
            root / "student",
            self.student,
            gold,
            self.ce_weight,
            self.epochs,
            self.batch_size,
            self.lr,
            device=pick_device(self.device),
            seed=self.seed,
            log=self.log,
            multi_label=task.multi_label,
            val_share=self.val_share,
            outside=task.outside_id,
        )
        export_onnx(root / "student", root / "onnx", int8=self.int8)
        shutil.rmtree(root / "student")
        self._work = work  # the exported model lives here until the Distiller is garbage collected
        self._fitted_task = task
        self._predictor = Predictor(root / "onnx")
        self.teacher_scores_ = np.asarray(scores, dtype=np.float32)
        self.classes_ = list(task.labels)
        return self

    @property
    def predictor(self):
        """The fitted model's :class:`~tinydistill.export.Predictor`."""
        p = getattr(self, "_predictor", None)
        if p is None:
            raise NotFittedError("this Distiller is not fitted yet: call fit(texts) or Distiller.load(path)")
        return p

    def predict(self, texts: list[str]) -> list:
        """The label id per text, or for a multi-label task the list of label ids with probability >= 0.5."""
        return self.predictor.predict(list(texts))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Label probabilities, one row per text and one column per label in ``classes_``."""
        return self.predictor.predict_proba(list(texts))

    def describe(self, texts: list[str]) -> list[dict]:
        """Per text, JSON-ready dicts as :meth:`~tinydistill.export.Predictor.describe` returns them."""
        return self.predictor.describe(list(texts))

    def score(self, texts: list[str], gold: list) -> float:
        """Accuracy of ``predict(texts)`` against ``gold`` (label ids, names or indices; for a multi-label
        task lists of them, and the exact-match accuracy: every label of a text right)."""
        from .evaluate import accuracy

        gold = self._gold(self._fitted_task, list(gold))
        if any(g is None for g in gold):
            raise ValueError("score needs a gold label for every text")
        index = {name: i for i, name in enumerate(self.classes_)}
        pred = self.predict(texts)
        if self._fitted_task.multi_label:
            return accuracy([{index[n] for n in p} for p in pred], [set(g) for g in gold])
        return accuracy([index[n] for n in pred], gold)

    def save(self, path: str | Path) -> Path:
        """Write the exported model to the folder ``path`` (created if needed), with the task as
        ``task.yaml``. :class:`~tinydistill.export.Predictor` and :meth:`load` read it.

        :returns: the folder.
        """
        import yaml

        src, out = self.predictor.file.parent, Path(path)
        if src.resolve() != out.resolve():
            shutil.copytree(src, out, dirs_exist_ok=True)
            if not (src / "model.int8.onnx").exists():  # an earlier save's, which Predictor would pick
                (out / "model.int8.onnx").unlink(missing_ok=True)
        (out / "task.yaml").write_text(
            yaml.safe_dump(self._fitted_task.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, path: str | Path, **params) -> Distiller:
        """A fitted Distiller for a saved or exported model (a folder, or ``hf:user/name``); the teacher is
        not needed to predict. The task comes from the folder's ``task.yaml``; a folder without one (such as
        a run's ``onnx`` folder) gets a task with only the model's labels. ``params`` set constructor
        arguments, for a later ``fit``."""
        from .export import Predictor, model_path

        folder = model_path(path)
        predictor = Predictor(folder)
        if (folder / "task.yaml").exists():
            task = load_task(folder / "task.yaml")
        else:
            labels = predictor.labels
            task = Task(folder.name, labels, labels, "{labels}\n{text}", multi_label=predictor.multi_label)
        d = cls(task, **params)
        d._fitted_task, d._predictor, d.classes_ = task, predictor, list(task.labels)
        return d
