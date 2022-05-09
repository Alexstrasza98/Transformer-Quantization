import torch
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader

# used to create key-data pair for map-style dataset
i = -1
def transformation(data):
    global i
    i += 1
    return (i, data)

def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    max_size = sequences[0].size()

class AG_NEWS_DATASET():
    def __init__(self, tokenizer=None, batch_size=32, max_sen_len =None):
        self.tokenizer = tokenizer
        self.specials = ['[UNK]', '[PAD]', '[CLS]']
        self.batch_size = batch_size
        self.max_sen_len = 100
        train_ds = AG_NEWS(root='./data', split='train')
        self.train_ds = train_ds.to_map_datapipe(key_value_fn=transformation)
        self.test_ds = AG_NEWS(root='./data', split='test')
        
    def generate_batch(self, data_batch):
        batch_sentence, batch_label = [], []
        for (lab, sen) in data_batch:
            batch_sentence.append(sen)
            batch_label.append(lab)
           
        tmp = self.tokenizer(batch_sentence, padding='longest', return_tensors='pt') 
        
        batch_sentence = tmp['input_ids']
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        attn_mask = tmp['attention_mask']
        
        return batch_sentence, batch_label, attn_mask
    
    def load_data(self):

        train_dl = DataLoader(self.train_ds, 
                              batch_size=self.batch_size, 
                              shuffle=True,
                              collate_fn=self.generate_batch)
        test_dl = DataLoader(self.test_ds,
                             batch_size=self.batch_size,
                             shuffle=True,
                             collate_fn=self.generate_batch)
        
        return train_dl, test_dl
