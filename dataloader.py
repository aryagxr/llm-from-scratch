from torch.utils.data import Dataset, DataLoader
import torch
import tiktoken

class GPTDatasetV1(Dataset): 
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = [] #input data ID
        self.target_ids = [] # target ID (to be predicted)

        # tokenize
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # sliding window to extract chunks of max length into overlapping sequences
        for i in range(0, len(token_ids)-max_length, stride): # subtracting max length to prevent out of bound error
            input_chunk = token_ids[i: i+max_length]
            target_chunk = token_ids[i+1: i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
            

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
        


def create_dataloader_1(txt, batch_size = 4, max_length = 256, stride=128, shuffle = True, num_workers = True, drop_last = True):
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0
    )

    return dataloader