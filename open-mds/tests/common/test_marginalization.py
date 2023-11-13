"""
Test our beamsearch generation
"""

import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from open_mds.common.generate import modified_generate, beam_search


def test_beam_search_n1_input() -> None:
    """
    We test the beam search, n=1 case (which is greedy), where we only have a single input document.
    This case should match the vanilla implementation.
    """

    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384-ms2")
    tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384-ms2")

    text = "hello world!"
    tokens = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**tokens, num_beams=1).squeeze().tolist()

    inputs = [
        {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
        }
    ]
    modified_generated_tokens = modified_generate(
        inputs=inputs,
        model=model,
        tokenizer=tokenizer,
        num_beams=1
    ).squeeze().tolist()

    assert generated_tokens == modified_generated_tokens

def test_beam_search_n2_input_different() -> None:
    """
    Here, we test the n=2 beam case.

    If there is only a single document, then the result should be unchanged.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384-ms2")
    tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384-ms2")

    text = "hello world!"
    tokens = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**tokens, num_beams=2).squeeze().tolist()
    inputs = [
        {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask']
        }
    ]
    modified_generated_tokens = modified_generate(
        inputs=inputs,
        model=model,
        tokenizer=tokenizer,
        num_beams=2
    ).squeeze().tolist()

    assert generated_tokens == modified_generated_tokens

def test_beam_search_n2_input_different_s2() -> None:
    """
    Here, we test the n=2 beam case. Except, we have two different 
    sets of documents. This is to sanity check that we are actually
    generating different output; "related" docs.
    """

    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384-ms2")
    tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384-ms2")

    text = "hello world!"
    tokens = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**tokens, num_beams=2).squeeze().tolist()
    inputs = [
        {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask']
        }
    ]
    text2 = "hello worlds, hi"
    tokens2 = tokenizer(text2, return_tensors="pt")
    inputs = [
        {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask']
        },
        {
            'input_ids': tokens2['input_ids'],
            'attention_mask': tokens2['attention_mask']
        }
    ]

    modified_generated_tokens = modified_generate(
        inputs=inputs,
        model=model,
        tokenizer=tokenizer,
        num_beams=2
    ).squeeze().tolist()

    assert generated_tokens == modified_generated_tokens

def test_beam_search_n2_input_different_s2_unrelated() -> None:
    """
    Here, we test the n=2 beam case. Except, we have two different 
    sets of documents. This is to sanity check that we are actually
    generating different output; "related" docs.
    """

    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384-ms2")
    tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384-ms2")

    text = "hello world!"
    tokens = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**tokens, num_beams=2).squeeze().tolist()
    inputs = [
        {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask']
        }
    ]
    text2 = "cats are the best pet"
    tokens2 = tokenizer(text2, return_tensors="pt")
    inputs = [
        {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask']
        },
        {
            'input_ids': tokens2['input_ids'],
            'attention_mask': tokens2['attention_mask']
        }
    ]

    modified_generated_tokens = modified_generate(
        inputs=inputs,
        model=model,
        tokenizer=tokenizer,
        num_beams=2
    ).squeeze().tolist()

    assert generated_tokens != modified_generated_tokens