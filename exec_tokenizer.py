from pprint import pprint
import textwrap

from langextract.core.tokenizer import tokenize


def research_tokenizer():
    txt = textwrap.dedent(
        """
        株式会社ハッチュウと株式会社ウケオイは、Webアプリケーション開発業務委託に関し、以下の通り契約を締結する。
        """
    )
    tokenized_text = tokenize(txt)
    pprint(tokenized_text)
    for token in tokenized_text.tokens:
        text = txt[token.char_interval.start_pos : token.char_interval.end_pos]
        print(f"text:{text}, type:{token.token_type}")


if __name__ == "__main__":
    research_tokenizer()
