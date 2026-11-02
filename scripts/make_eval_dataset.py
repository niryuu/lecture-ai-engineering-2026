"""第2回 宿題（発展）: 用語集から評価用の質問と正解（JSONL）を LLM で合成する。

使い方:
    uv run python scripts/make_eval_dataset.py > glossary_eval.jsonl

OpenAI API 互換のエンドポイントを使う（既定: http://localhost:8000/v1、モデル Muse-Glimmer-30B）。
環境変数 EVAL_BASE_URL / EVAL_MODEL で変えられる。

1 行 = 1 問: {"question", "kind", "term_id", "must_include"}
- seen       : 学習データと同じ質問（テンプレートから作る。LLM は使わない）
- alias      : 別名で聞く質問（テンプレートから作る）
- paraphrase : 言い回しを変えた質問（LLM が作る）
- unknown    : 用語集にない語の質問（LLM が語と質問を作る）
must_include は、用語の質問では LLM が定義から抜き出した語句（定義に含まれるものだけ残す）、
unknown では「用語集にありません」。
"""

import argparse
import json
import os
import random
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_sft_dataset import GLOSSARY, TEMPLATES  # noqa: E402

BASE_URL = os.environ.get("EVAL_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("EVAL_MODEL", "Muse-Glimmer-30B")
UNKNOWN_PHRASE = "用語集にありません"
MAX_KEYWORD_LEN = 15


def chat(prompt: str) -> str:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": 4096}
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as res:
        content = json.load(res)["choices"][0]["message"]["content"] or ""
    content = content.split("<|message|>")[-1]  # <|channel|> final<|message|> 形式なら最後の本文だけ使う
    return re.sub(r"<think>.*?</think>", "", content, flags=re.S)  # <think> 形式の思考部分も除く


def ask_json(prompt: str) -> dict:
    """JSON を返すまで 3 回まで聞き直す"""
    for _ in range(3):
        m = re.search(r"\{.*\}", chat(prompt), re.S)
        try:
            return json.loads(m.group(0))
        except (AttributeError, json.JSONDecodeError):
            print("JSON が読めなかったので再試行します", file=sys.stderr)
    raise RuntimeError("LLM から JSON を得られませんでした")


def term_prompt(t: dict, n: int) -> str:
    return f"""次は講座の用語集の 1 項目です。
用語: {t["term"]}
別名: {"、".join(t["aliases"])}
定義: {t["definition"]}

評価データとして、次の 2 つを作ってください。
1. questions: この用語の意味を尋ねる日本語の質問を {n} 個。「{t["term"]}」という用語名を必ずそのまま含め、「〜とは何ですか？」「〜について簡単に説明してください。」以外の言い回しにする。
2. must_include: 正しい回答に必ず含まれるべき短い語句（10 文字以内）を 1 個だけ。定義の文中から一字一句そのまま抜き出す。一般常識だけでは出てこない、この定義に特有の語句を優先する。

JSON だけを返してください: {{"questions": ["..."], "must_include": ["..."]}}（must_include の要素は 1 個）"""


def unknown_prompt(n: int) -> str:
    return f"""AI・LLM・ソフトウェア開発に関する専門用語を {n} 個挙げ、それぞれの意味を尋ねる日本語の質問を 1 つずつ作ってください。
質問には用語をそのまま含めてください。

JSON だけを返してください: {{"items": [{{"term": "...", "question": "..."}}]}}"""


def in_glossary(word: str, terms: list[dict]) -> bool:
    w = word.lower()
    keys = [k.lower() for t in terms for k in [t["term"], *t["aliases"]] if len(k) >= 2]
    return any(k in w or w in k for k in keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-terms", type=int, default=12, help="評価に使う用語の数")
    ap.add_argument("--n-paraphrase", type=int, default=1, help="用語ごとの言い換え質問の数")
    ap.add_argument("--n-unknown", type=int, default=12, help="用語集にない語の質問の数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    terms = json.loads(GLOSSARY.read_text(encoding="utf-8"))["terms"]
    train_questions = {tpl.format(term=t["term"]) for t in terms for tpl in TEMPLATES}
    rows = []

    for t in random.Random(args.seed).sample(terms, args.n_terms):
        print(f"用語: {t['term']}", file=sys.stderr)
        out = ask_json(term_prompt(t, args.n_paraphrase))
        must = [w for w in out.get("must_include", []) if w and w in t["definition"] and len(w) <= MAX_KEYWORD_LEN][:1]  # 定義に無い・長すぎる語句は捨て、1 個だけ使う
        if not must:
            print(f"  must_include が定義に見つからないのでスキップ: {out.get('must_include')}", file=sys.stderr)
            continue
        base = {"term_id": t["id"], "must_include": must}
        rows.append({"question": TEMPLATES[0].format(term=t["term"]), "kind": "seen", **base})
        rows.append({"question": f"「{t['aliases'][0]}」はどういう意味ですか？", "kind": "alias", **base})
        names = [t["term"], t["term"].split("（")[0], *t["aliases"]]
        rows += [{"question": q, "kind": "paraphrase", **base}  # 用語名を含まない質問・学習データと同じ質問は捨てる
                 for q in out.get("questions", [])[: args.n_paraphrase] if any(k in q for k in names) and q not in train_questions]

    print("用語集にない語", file=sys.stderr)
    items = ask_json(unknown_prompt(args.n_unknown * 2))["items"]  # 用語集と重なる語を捨てるので多めに頼む
    kept = [i for i in items if not in_glossary(i["term"], terms)
            and i["question"] not in train_questions]
    rows += [{"question": i["question"], "kind": "unknown", "term_id": None, "must_include": [UNKNOWN_PHRASE]}
             for i in kept[: args.n_unknown]]

    questions = [r["question"] for r in rows]
    rows = [r for i, r in enumerate(rows) if r["question"] not in questions[:i]]  # 重複を除く
    for r in rows:
        sys.stdout.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {k: sum(r["kind"] == k for r in rows) for k in ["seen", "alias", "paraphrase", "unknown"]}
    print(f"作成: {counts}", file=sys.stderr)


if __name__ == "__main__":
    main()
