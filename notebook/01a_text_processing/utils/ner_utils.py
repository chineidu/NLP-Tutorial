import json
from typing import Any, Dict, List

from rich.console import Console
from rich.theme import Theme

custom_theme = Theme(
    {
        "info": "#76FF7B",
        "warning": "#FBDDFE",
        "error": "#FF0000",
    }
)
console = Console(theme=custom_theme)


def _convert_to_iob(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Convert a single data item to IOB format.

    Parameters
    ----------
    data : Dict[str, Any]
        A dictionary containing 'text' and 'label' keys.

    Returns
    -------
    Dict[str, List[str]]
        A dictionary with 'tokens' and 'iob_tags' keys, both containing lists of strings.
    """
    id: str = data["id"]
    text: str = data["text"]
    labels: List[List[Any]] = data["label"]

    tokens: List[str] = text.split()
    iob_tags: List[str] = ["O"] * len(tokens)

    for label in labels:
        try:
            start: int
            end: int
            entity: str
            start, end, entity = label
            start_token_idx: int = len(text[:start].split())
            end_token_idx: int = len(text[:end].split())
        except Exception as error:
            console.print(f"{id = } | {error = }")
            continue

        if start_token_idx == end_token_idx:
            iob_tags[start_token_idx] = f"B-{entity}"
        else:
            iob_tags[start_token_idx] = f"B-{entity}"
            for i in range(start_token_idx + 1, end_token_idx):
                iob_tags[i] = f"I-{entity}"

    return {"tokens": tokens, "iob_tags": iob_tags}


def convert_to_iob(data: List[Dict[str, Any]], filepath: str = "output.jsonl") -> None:
    """
    Convert a list of data items to IOB format and write to a JSONL file.

    Parameters
    ----------
    data : List[Dict[str, Any]]
        A list of dictionaries, each containing 'text' and 'label' keys.
    filepath : str, optional
        The path to the output file (default is "output.jsonl").

    Returns
    -------
    None
    """
    try:
        data_list: List[Dict[str, List[str]]] = [_convert_to_iob(d) for d in data]
    except Exception as error:
        console.print(f"{error = }")
        return

    finally:
        console.print(f"{len(data_list) = }")
        convert_to_jsonl(data_list, filepath)


def convert_to_jsonl(data: List[Dict[str, List[str]]], filepath: str = "output.jsonl") -> None:
    """
    Convert data to JSONL format and write to a file.

    Parameters
    ----------
    data : List[Dict[str, List[str]]]
        A list of dictionaries, each containing 'tokens' and 'iob_tags' keys with
        lists of strings.
    filepath : str, optional
        The path to the output file (default is "output.jsonl").

    Returns
    -------
    None
    """
    with open(filepath, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")
