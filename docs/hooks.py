"""MkDocs hook: fill `<!-- cli-reference -->` with the `--help` text of every tinydistill command, taken from
the real argument parser at build time, so the reference never drifts from the code."""

import os


def _reference() -> str:
    from tinydistill.cli import build_parser

    os.environ["COLUMNS"] = "100"  # argparse wraps help text to the terminal width
    parser = build_parser()
    sub = next(a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction")
    helps = {c.dest: c.help for c in sub._choices_actions}
    parts = ["Every command and option, from `tinydistill <command> --help`.\n"]
    parts.append("| Command | What it does |\n|---|---|")
    parts += [f"| [`{name}`](#{name}) | {helps.get(name, '')} |" for name in sub.choices]
    for name, cmd in sub.choices.items():
        usage = cmd.format_help().rstrip()
        parts.append(f"\n## {name}\n\n{helps.get(name, '').capitalize()}.\n\n```text\n{usage}\n```")
    return "\n".join(parts) + "\n"


def on_page_markdown(markdown: str, **_) -> str:
    if "<!-- cli-reference -->" in markdown:
        return markdown.replace("<!-- cli-reference -->", _reference())
    return markdown
