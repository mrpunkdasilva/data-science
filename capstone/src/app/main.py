"""Entry point of the CSDS-352 capstone: an interactive document knowledge base."""

from textwrap import dedent

from lib.console_manager import console_manager

HELLO_WORLD = dedent(
    """
    # Machine Learning Vault

    Welcome to your **machine learning** vault! This is the capstone entry point.

    The project is an interactive application to explore a corpus of scientific
    papers using **embeddings** and **clusterization**. It covers the four
    requirements of the brief:

    - **Ingestion**: papers are fetched from the arXiv API as plain text or PDF
    - **Knowledge base**: embeddings are stored in a local NumPy vector store
    - **Exploration**: a Streamlit UI renders the cluster map and semantic search
    - **Quality control**: quality scoring plus anomaly detection

    ## How to use black and mypy

    ```bash

    $ black .
    All done! ✨ 🍰 ✨
    N files left unchanged.

    $ mypy .
    Success: no issues found in N source files

    ```

    ## How to verify if my plain files are in compliance

    ```bash

    $ ./ci/scripts/check-ipynb-compliance.sh
    ...
    All notebooks are compliant (no outputs found)!

    $ ./ci/scripts/check-plain-compliance.sh
    ...
    All .typ, .txt, .sh, .md, and .yml files are compliant!

    ```

    ## How to run the capstone

    ```bash

    $ pip install -e .
    $ capstone ingest --max-results 500
    $ capstone build --backend tfidf
    $ capstone cluster --n-clusters 8
    $ capstone search "how do neural networks learn representations"
    $ capstone serve

    ```
    """
)


def main() -> None:
    console_manager.print_markdown(HELLO_WORLD)


if __name__ == "__main__":
    main()
