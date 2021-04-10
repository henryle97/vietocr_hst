class Vocab():
    def __init__(self, chars):
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.mask_token = 3

        self.chars = chars

        self.c2i = {c:i+4 for i, c in enumerate(chars)}

        self.i2c = {i+4:c for i, c in enumerate(chars)}
        
        self.i2c[0] = '<pad>'
        self.i2c[1] = '<sos>'
        self.i2c[2] = '<eos>'
        self.i2c[3] = '*'

    def encode(self, chars):
        return [self.go] + [self.c2i[c] for c in chars] + [self.eos]
    
    def decode(self, ids):
        first = 1 if self.go in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        sent = ''.join([self.i2c[i] for i in ids[first:last]])
        return sent

    def decode_crnn(self, ids):
        first = 1 if self.go in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        char_list = []
        for index, id_char in enumerate(ids[first:last]):
            if id_char != 0 and (not (index > 0 and ids[index-1] == ids[index])):
                char_list.append(id_char)
        return ''.join([self.i2c[i]  for i in char_list])
    
    def __len__(self):
        return len(self.c2i) + 4
    
    # def batch_decode(self, arr):
    #     texts = [self.decode(ids) for ids in arr]
    #     return texts

    def batch_decode(self, arr, crnn=False):
        if crnn:
            texts = [self.decode_crnn(ids) for ids in arr]
        else:
            texts = [self.decode(ids) for ids in arr]
        return texts

    def __str__(self):
        return self.chars
