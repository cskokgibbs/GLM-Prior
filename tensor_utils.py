import torch

from typing import Dict, List


def collate_tensors(
    tensor_dicts: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Given a list of tensor dictionaries, each containing the fields 'input_ids'
    and 'attention_mask', pad and/or truncate all tensors to the same length and stack.
    """
    if not tensor_dicts:
        return {}
    max_tensor_len = int(
        max([torch.sum(d["attention_mask"]).item() for d in tensor_dicts])
    )
    for d in tensor_dicts:
        if len(d["input_ids"]) < max_tensor_len:
            remaining_len = max_tensor_len - len(d["input_ids"])
            d["input_ids"] = torch.cat(
                [
                    d["input_ids"],
                    torch.full(
                        (remaining_len,),
                        fill_value=pad_token_id,
                        dtype=torch.long,
                    ).to(d["input_ids"].device),
                ]
            )
            d["attention_mask"] = torch.cat(
                [
                    d["attention_mask"],
                    torch.zeros(
                        (remaining_len,),
                        dtype=torch.long,
                    ).to(d["attention_mask"].device),
                ]
            )
        elif len(d["input_ids"]) > max_tensor_len:
            d["input_ids"] = d["input_ids"][:max_tensor_len]
            d["attention_mask"] = d["attention_mask"][:max_tensor_len]
    return {
        "input_ids": torch.stack([d["input_ids"] for d in tensor_dicts]),
        "attention_mask": torch.stack([d["attention_mask"] for d in tensor_dicts]),
    }


def remove_padding(
    tensor_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Given a tensor dictionary containing the fields 'input_ids'
    and 'attention_mask', remove excess padding.
    """
    max_len = torch.sum(tensor_dict["attention_mask"]).item()
    tensor_dict["input_ids"] = tensor_dict["input_ids"][:max_len]
    tensor_dict["attention_mask"] = tensor_dict["attention_mask"][:max_len]
    return tensor_dict
