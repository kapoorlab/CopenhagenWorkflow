#!/usr/bin/env python3
"""
llm_doc_updater.py

Read a Python file or directory of .py files, send each file's contents to OpenAI GPT to:
 1. Strip ALL existing docstrings under any def or class.
 2. Generate fresh NumPy-style docstrings for every function, method, and __init__.
 3. Preserve all other code and formatting exactly.

Respond ONLY with the fully rewritten Python source (no markdown fences).

Usage:
  export OPENAI_API_KEY="sk-..."
  python llm_doc_updater.py path/to/file_or_dir.py [--model gpt-4]
"""
import os
import sys
import argparse
import openai


def update_file_with_llm(path: str, model: str):
    """
    Read the file at `path`, send to OpenAI for docstring refresh, and overwrite it.
    """
    with open(path, 'r', encoding='utf-8') as f:
        source = f.read()

    prompt = f"""
You are an expert Python refactoring assistant.

Given this Python source, please:
1. Remove ALL existing docstrings under any def or class.
2. For each function, method, and __init__, add a new NumPy-style docstring with a one-sentence summary, Parameters and Returns sections.
3. Preserve all other code and formatting exactly.

Source:
{source}
"""

    response = openai.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
    )
    updated_source = response.choices[0].message.content

    with open(path, 'w', encoding='utf-8') as f:
        f.write(updated_source)
    print(f"Updated docstrings in: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-based automatic docstring remover and rewriter"
    )
    parser.add_argument('path', help=".py file or directory to process")
    parser.add_argument('--model', default='gpt-4', help="OpenAI model to use")
    args = parser.parse_args()

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: Please set the OPENAI_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)
    openai.api_key = api_key

    targets = []
    if os.path.isdir(args.path):
        for root, _, files in os.walk(args.path):
            for name in files:
                if name.endswith('.py'):
                    targets.append(os.path.join(root, name))
    elif args.path.endswith('.py') and os.path.isfile(args.path):
        targets.append(args.path)
    else:
        print("Error: Must specify a .py file or directory", file=sys.stderr)
        sys.exit(1)

    for filepath in targets:
        update_file_with_llm(filepath, args.model)

if __name__ == '__main__':
    main()
