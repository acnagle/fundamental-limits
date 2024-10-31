import torch

def get_input(context, query, answer, tokenizer, force_tokenization=False, with_instructions=True, query_first=False, get_item=False, device='cpu'):
    if with_instructions:    # if we want the instruciton fine-tuning dataset
        if force_tokenization:
            if query_first:     # LLMLingua Query
                context_ids = torch.cat([tokenizer(context[i], add_special_tokens=False, return_tensors='pt')['input_ids'][0] for i in range(len(context))], dim=0)
                input_ids = torch.cat([torch.tensor([tokenizer.bos_token_id]), tokenizer(query + "\n\n", add_special_tokens=False, return_tensors='pt')['input_ids'][0], context_ids, tokenizer(" " + answer, add_special_tokens=False, return_tensors='pt')['input_ids'][0]], dim=0).to(device)      # when query is first we assume we are training the encoder model, so we do not provide the answer
            else:
                context_ids = torch.cat([tokenizer(context[i], add_special_tokens=False, return_tensors='pt')['input_ids'][0] for i in range(len(context))], dim=0)
                input_ids = torch.cat([torch.tensor([tokenizer.bos_token_id]), context_ids, tokenizer(" " + query  + " " + answer, add_special_tokens=False, return_tensors='pt')['input_ids'][0]], dim=0).to(device)
            attention_mask = torch.ones_like(input_ids)
        else:
            if query_first:     # LLMLingua Query
                tokenized_example = tokenizer(query + "\n\n" + context + " " + answer, return_attention_mask=True, return_tensors='pt')     # "\n\n" since LLMLingua Query uses it
            else:
                tokenized_example = tokenizer(context + " " + query + " " + answer, return_attention_mask=True, return_tensors='pt')
            # check if the first token is the bos token, if not add it
            if tokenized_example['input_ids'][0][0] != tokenizer.bos_token_id:
                tokenized_example['input_ids'] = torch.cat([torch.tensor([tokenizer.bos_token_id]), tokenized_example['input_ids'][0]], dim=0)
                tokenized_example['attention_mask'] = torch.cat([torch.ones(1), tokenized_example['attention_mask'][0]], dim=0)
            input_ids = tokenized_example['input_ids'].squeeze().to(device)
            attention_mask = tokenized_example['attention_mask'].squeeze().to(device)

        labels = input_ids.clone()  # NOTE: when query is first, we will just do auto-regressive training
        if not query_first:
            labels[:-1] = -100
    else:   # if we just want the sequences next token prediction
        if force_tokenization:
            context_ids = torch.cat([tokenizer(context[i], add_special_tokens=False, return_tensors='pt')['input_ids'][0] for i in range(len(context))], dim=0)
            input_ids = torch.cat([torch.tensor([tokenizer.bos_token_id]), context_ids]).to(device)
            attention_mask = torch.ones_like(input_ids)
        else:
            tokenized_example = tokenizer(context, return_attention_mask=True, add_special_tokens=True, return_tensors='pt')
            # check if the first token is the bos token, if not add it
            if tokenized_example['input_ids'][0][0] != tokenizer.bos_token_id:
                tokenized_example['input_ids'] = torch.cat([torch.tensor([tokenizer.bos_token_id]), tokenized_example['input_ids'][0]], dim=0)
                tokenized_example['attention_mask'] = torch.cat([torch.ones(1), tokenized_example['attention_mask'][0]], dim=0)
            input_ids = tokenized_example['input_ids'].squeeze().to(device)
            attention_mask = tokenized_example['attention_mask'].squeeze().to(device)

        labels = input_ids.clone()

    if not get_item:        # if we're not using this function in a Dataset class
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        labels = labels.unsqueeze(0)

    return input_ids, attention_mask, labels
