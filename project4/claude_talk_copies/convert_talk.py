"""
Claude会話ログをMarkdown形式に変換するスクリプト

使い方:
  python convert_talk.py <jsonl_file> <output_md_file>
  python convert_talk.py --txt <txt_file> <output_md_file>
"""

import json
import sys
import re
from datetime import datetime, timezone


def txt_to_md(txt_path, md_path):
    """既存のtxt（ターミナルコピー形式）をMarkdownに変換"""
    with open(txt_path, encoding="utf-8") as f:
        raw = f.read()

    # ❯ で始まる行 = ユーザー発言、● で始まる行 = Claude発言
    # テキストを分割してmarkdownに変換
    lines = raw.splitlines()
    blocks = []
    current_role = None
    current_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("❯"):
            if current_lines and current_role:
                blocks.append((current_role, "\n".join(current_lines).strip()))
            current_role = "user"
            content = re.sub(r"^❯\s*", "", stripped)
            current_lines = [content] if content else []
        elif stripped.startswith("●"):
            if current_lines and current_role:
                blocks.append((current_role, "\n".join(current_lines).strip()))
            current_role = "assistant"
            content = re.sub(r"^●\s*", "", stripped)
            current_lines = [content] if content else []
        else:
            if current_role:
                current_lines.append(line.rstrip())

    if current_lines and current_role:
        blocks.append((current_role, "\n".join(current_lines).strip()))

    # Markdown出力
    date_str = md_path.stem if hasattr(md_path, "stem") else "unknown"
    out_lines = [f"# Claude Talk — {date_str}\n"]

    for role, content in blocks:
        if role == "user":
            out_lines.append("## User\n")
        else:
            out_lines.append("## Claude\n")

        # コードブロック検出（インデント4スペースまたは既存のコードブロック）
        content = content.strip()
        out_lines.append(content + "\n")
        out_lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"Saved: {md_path}")


def extract_text(content):
    """message.contentからテキスト部分のみ抽出（tool_use/tool_resultは除外）"""
    if isinstance(content, str):
        return content, False
    if isinstance(content, list):
        text_parts = []
        has_text = False
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "").strip()
                if t:
                    text_parts.append(t)
                    has_text = True
        return "\n".join(text_parts), has_text
    return "", False


def jsonl_to_md(jsonl_path, md_path, date_str=None):
    """JWSONLをMarkdownに変換（テキストのみ、ツール呼び出しは省略）"""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    messages = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = obj.get("type")
            if msg_type not in ("user", "assistant"):
                continue

            message = obj.get("message", {})
            role = message.get("role", msg_type)
            content = message.get("content", "")
            text, has_text = extract_text(content)

            if not has_text:
                continue  # ツール呼び出しのみのメッセージはスキップ

            # 直前と同じroleの場合は連結（ツール結果を挟んだ分割を統合）
            if messages and messages[-1][0] == role:
                messages[-1] = (role, messages[-1][1] + "\n\n" + text.strip())
            else:
                messages.append((role, text.strip()))

    out_lines = [f"# Claude Talk — {date_str}\n"]
    for role, text in messages:
        if role == "user":
            out_lines.append("## User\n")
        else:
            out_lines.append("## Claude\n")
        out_lines.append(text + "\n")
        out_lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"Saved: {md_path}")
    print(f"  {len(messages)} messages")


if __name__ == "__main__":
    import pathlib

    args = sys.argv[1:]

    if "--txt" in args:
        idx = args.index("--txt")
        txt_path = pathlib.Path(args[idx + 1])
        out_path = pathlib.Path(args[idx + 2])
        txt_to_md(txt_path, out_path)
    elif len(args) >= 2:
        jsonl_path = pathlib.Path(args[0])
        out_path   = pathlib.Path(args[1])
        date_str   = args[2] if len(args) >= 3 else out_path.stem
        jsonl_to_md(jsonl_path, out_path, date_str)
    else:
        print(__doc__)
