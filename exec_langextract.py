import os
import pprint

import langextract as lx
import textwrap


prompt = """
契約書から取引先の会社名を抽出してください。
自社名は「株式会社ハッチュウ」です。
"""

examples = [
    lx.data.ExampleData(
        text=textwrap.dedent(
            """\
            株式会社ハッチュウと株式会社ウケオイは、
            Webアプリケーション開発業務委託に関し、以下の通り契約を締結する。"""
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="取引先",
                extraction_text="株式会社ウケオイ",
                attributes=None
            )
        ]
    )
]

result = lx.extract(
    text_or_documents=textwrap.dedent(
        """\
        株式会社ハッチュウと株式会社ヤリマスは、新商品デザイン制作に関し、以下の通り契約を締結する。"""
    ),
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash"
)

pprint.pprint(result)

lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")

# Generate the visualization from the file
html_content = lx.visualize("extraction_results.jsonl")
with open("visualization.html", "w") as f:
    if hasattr(html_content, 'data'):
        f.write(html_content.data)  # For Jupyter/Colab
    else:
        f.write(html_content)
