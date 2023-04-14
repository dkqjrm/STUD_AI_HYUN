from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm


class StudDataset(Dataset):
    def __init__(self, path):
        self.preprocessing(path)
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        text = self.text_data[idx]
        # lecture = self.lecture_data[idx]
        label = self.label_data[idx]
        input_data = self.tokenizer.encode_plus(text,
                                                return_tensors='pt',
                                                padding='max_length',
                                                truncation=True)
        input_data.update({"label": torch.tensor(label)})
        input_data = {k: v.squeeze() for k, v in input_data.items()}

        return input_data

    def preprocessing(self, path):
        df = pd.read_csv(path, encoding='utf-8-sig', sep='\t')
        df = df.dropna()
        self.text_data = []
        # self.lecture_data = []
        self.label_data = []
        for seq in tqdm(df.iloc, total=df.shape[0], desc="preprocessing"):
            tmp = list(seq)
            self.text_data.append(str(tmp[0])+' '+str(tmp[1])+' '+str(tmp[1]))
            # self.lecture_data.append(str(tmp[2]))
            self.label_data.append(int(tmp[3]))

    def custom_collate(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        label = torch.stack([item['label'] for item in batch])
        # labels = torch.stack([item['labels'] for item in batch])

        # Calculate max length of sequences in batch
        max_length = torch.max(torch.sum(attention_mask, dim=1)).item()

        return {'input_ids': input_ids[:, :max_length],
                'token_type_ids' : token_type_ids[:, :max_length],
                'attention_mask': attention_mask[:, :max_length],
                'label': label}
                # 'labels': labels}

if __name__ == "__main__":
    dataset = StudDataset('../val.tsv')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, collate_fn=dataset.custom_collate)

    for i in dataloader:
        # print(i)
        print({k: v.shape for k, v in i.items()})
        print({k: v for k, v in i.items()})
        break
