import torch
from transformers import BeamSearchScorer

def modified_beam_search(
    inputs,
    model,
    tokenizer,
    # beam_scorer,
    logits_processor,
    num_beams=1,
    device="cpu"
):
    """
    Beam search modified for multiple documents.
    """
    # instantiate beam scorer
    beam_scorer = BeamSearchScorer(
        batch_size=1,
        num_beams=num_beams,
        device=model.device,
    )
    model_kwargs_list = []

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams
    batch_beam_size, cur_len = inputs[0]['input_ids'].shape

    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
    # beam_scores[:, 1:] = -1e9
    beam_scores[:, 1:] = float('-inf')
    beam_scores = beam_scores.view((batch_size * num_beams,))

    inputs_list = [i['input_ids'].repeat(num_beams, 1).to(device) for i in inputs]
    atten_list = [i['attention_mask'].repeat(num_beams, 1).to(device) for i in inputs]
    generated_sequence = torch.tensor([[tokenizer.eos_token_id]]).repeat(num_beams, 1).to(device)  # initial token

    # print("Here's generated_sequence:", generated_sequence)
    # We cache the initial information so we don't have to repeat the calculations every time.
    # Do this first initially
    # all_past_key_values = len(inputs_list) * [None]
    # all_encoder_outputs = []
    probs_list = []
    outputs_list = []
    for idx, (new_input, new_atten) in enumerate(zip(inputs_list, atten_list)):
        # print("Here's new input:", tokenizer.batch_decode(new_input))
        # print("New input length:", new_input.shape)
        # print("Here's generated_sequence:", generated_sequence)
        with torch.no_grad():
            model_kwargs = {
                    "encoder_outputs": model.get_encoder()(
                        new_input.repeat_interleave(num_beams, dim=0), return_dict=True
                    )
            }  # Should give me N beams per input document
            model_kwargs_list.append(model_kwargs)
            # print("Here's model_kwargs:", model_kwargs)

            model_inputs = model.prepare_inputs_for_generation(generated_sequence, **model_kwargs)

            outputs = model(
                **model_inputs,
                return_dict=True,
            )
            # outputs = model(
            #     **model_inputs,
            #     input_ids=new_input,
            #     attention_mask=new_atten,
            #     decoder_input_ids=generated_sequence,
            #     past_key_values=None
            # )
            # breakpoint()
            next_token_logits = outputs.logits[:, -1, :]
            next_token_scores = next_token_logits.softmax(dim=1)
            next_token_scores_processed = logits_processor(new_input, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)
            # breakpoint()
            probs_list.append(next_token_scores)
            outputs_list.append(outputs)
            
        # all_encoder_outputs.append(outputs.encoder_last_hidden_state)
        # all_past_key_values[idx] = outputs.past_key_values
    
    # Average the probs list, then take token with highest probability
    # TODO: fix this for the single document case
    # TODO: remove "next_token"
    # if len(probs_list) == 1:
    #     next_token = probs_list[0].argmax()
    # else:
    #     average_probs = torch.mean(torch.stack(probs_list).squeeze(), dim=0)
    #     next_token = average_probs.argmax()
    average_probs = torch.mean(torch.stack(probs_list).squeeze(), dim=0)

    # After averaging, reshape and check
    vocab_size = average_probs.shape[-1]
    average_probs_scores = average_probs.view(batch_size, num_beams * vocab_size)

    # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
    average_probs_scores, next_tokens = torch.topk(
        average_probs_scores, 2 * num_beams, dim=1, largest=True, sorted=True
    )

    next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
    next_tokens = next_tokens % vocab_size

    beam_outputs = beam_scorer.process(
        generated_sequence,
        average_probs_scores,
        next_tokens,
        next_indices,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        beam_indices=None 
    )

    beam_scores = beam_outputs["next_beam_scores"]
    beam_next_tokens = beam_outputs["next_beam_tokens"]
    beam_idx = beam_outputs["next_beam_indices"]
    generated_sequence = torch.cat([generated_sequence[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
    model_kwargs_list = [
        model._update_model_kwargs_for_generation(
            outputs_list[idx], kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        ) for idx, kwargs in enumerate(model_kwargs_list)
    ]
    # model_kwargs = model._update_model_kwargs_for_generation(
    #     outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
    # )

    for idx, kwargs in enumerate(model_kwargs_list):
        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = model._reorder_cache(model_kwargs["past_key_values"], beam_idx)
    
    # Need to fix this?????
    # if model_kwargs["past_key_values"] is not None:
    #     model_kwargs["past_key_values"] = model._reorder_cache(model_kwargs["past_key_values"], beam_idx)

    cur_len = cur_len + 1

    # Append token to generated sequence
    with torch.no_grad():
        while True:
            probs_list = []
            outputs_list = []
            for idx, (new_input, new_atten) in enumerate(zip(inputs_list, atten_list)):
                # breakpoint()
                with torch.no_grad():
                    model_inputs = model.prepare_inputs_for_generation(generated_sequence, model_kwargs_list[idx])
                    outputs = model(
                        decoder_input_ids=generated_sequence,
                        encoder_outputs=model_kwargs_list[idx]['encoder_outputs'],
                        past_key_values=model_kwargs_list[idx]['past_key_values'],
                        return_dict=True
                        # decoder_input_ids=generated_sequence,
                        # past_key_values=all_past_key_values[idx],
                        # encoder_outputs=all_encoder_outputs[idx]
                    )
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_scores = next_token_logits.softmax(dim=1)
                    next_token_scores_processed = logits_processor(generated_sequence, next_token_scores)
                    next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)
                    # breakpoint()
                    probs_list.append(next_token_scores)
                    outputs_list.append(outputs)

            # Average the probs list, then take token with highest probability
            # if len(probs_list) == 1:
            #     next_token = probs_list[0].argmax()
            # else:
            #     average_probs = torch.mean(torch.stack(probs_list).squeeze(), dim=0)
            #     next_token = average_probs.argmax()
            average_probs = torch.mean(torch.stack(probs_list).squeeze(), dim=0)

            # After averaging, reshape and check
            vocab_size = average_probs.shape[-1]
            average_probs_scores = average_probs.view(batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            average_probs_scores, next_tokens = torch.topk(
                average_probs_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size
            # print("Batch decoding:", tokenizer.batch_decode(next_tokens))
            beam_outputs = beam_scorer.process(
                generated_sequence,
                average_probs_scores,
                next_tokens,
                next_indices,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                beam_indices=None 
            )

            # breakpoint()
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            generated_sequence = torch.cat([generated_sequence[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            # model_kwargs = model._update_model_kwargs_for_generation(
            #     outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
            # )

            # # Need to fix this?????
            # if model_kwargs["past_key_values"] is not None:
            #     model_kwargs["past_key_values"] = model._reorder_cache(model_kwargs["past_key_values"], beam_idx)

            model_kwargs_list = [
                model._update_model_kwargs_for_generation(
                    outputs_list[idx], kwargs, is_encoder_decoder=model.config.is_encoder_decoder
                ) for idx, kwargs in enumerate(model_kwargs_list)
            ]
            # model_kwargs = model._update_model_kwargs_for_generation(
            #     outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
            # )

            for idx, kwargs in enumerate(model_kwargs_list):
                if model_kwargs["past_key_values"] is not None:
                    model_kwargs["past_key_values"] = model._reorder_cache(model_kwargs["past_key_values"], beam_idx)
            
            cur_len = cur_len + 1

            # Append token to generated sequence
            # generated_sequence = torch.cat((generated_sequence, next_token.unsqueeze(0).unsqueeze(0)), dim=1)

            # Current stop cond is eos token id, need to get all eos for all
            # breakpoint()
            if beam_scorer.is_done:
                break

    sequence_outputs = beam_scorer.finalize(
        generated_sequence,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # max_length=stopping_criteria.max_length,
        max_length=None,
        beam_indices=None,
    )

    return sequence_outputs