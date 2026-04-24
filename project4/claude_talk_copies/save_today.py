"""
当日の会話をMarkdownに保存する。

使い方（Claude Code のプロンプトから）:
  ! python project4/claude_talk_copies/save_today.py
"""

import os
import glob
import pathlib
from datetime import datetime

PROJECT_JSONL_DIR = r"C:/Users/cs16/.claude/projects/C--Users-cs16-Roughness"
OUTPUT_DIR        = r"C:/Users/cs16/Roughness/project4/claude_talk_copies"

# 最新の .jsonl を自動選択
jsonl_files = glob.glob(os.path.join(PROJECT_JSONL_DIR, "*.jsonl"))
if not jsonl_files:
    print("ERROR: .jsonl ファイルが見つかりません")
    exit(1)

latest_jsonl = max(jsonl_files, key=os.path.getmtime)

date_str    = datetime.now().strftime("%Y%m%d")
date_label  = datetime.now().strftime("%Y-%m-%d")
output_path = pathlib.Path(OUTPUT_DIR) / f"{date_str}.md"

# 変換実行
import sys
sys.path.insert(0, OUTPUT_DIR)
from convert_talk import jsonl_to_md

jsonl_to_md(pathlib.Path(latest_jsonl), output_path, date_label)
print(f"JSONL: {latest_jsonl}")
