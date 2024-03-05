#from transformer import build_transformer_model
from functions import words_to_numbers,return_dict,return_order,split_list,number_to_words
from transformer import build_transformer_model
import numpy as np
from dictionary import dictionary


def train_model(dictionary_train: dict, maxlen: int,per,embed_dim,num_heads,ff_dim,
                num_blocks,dropout_rate,num_encoders,num_decoders,batch_size,
                epochs):
    dictionary_k = list(dictionary_train.keys())
    dictionary_v = list(dictionary_train.values())
    vocab = set([])

    for i in range(len(dictionary_k)):
        o = dictionary_k[i].split(' ')
        for j in range(len(o)):
            vocab.add(o[j])
    for i in range(len(dictionary_v)):
        o = dictionary_v[i].split(' ')
        for j in range(len(o)):
            vocab.add(o[j])

    vocab_list = sorted(list(vocab))
    d = return_dict(unique_words=vocab_list)
    x = []
    for i in range(len(dictionary_k)):
        o = dictionary_k[i].split(' ')
        x1 = [vocab_list.index(w) if w in vocab_list else 0 for w in o]
        x.append(x1)
    y = []
    for i in range(len(dictionary_v)):
        o = dictionary_v[i].split(' ')
        y1 = [vocab_list.index(w) if w in vocab_list else 0 for w in o]
        y.append(y1)




    x_ = []
    for seq in x:
        padded_seq = seq[:maxlen] + [0] * (maxlen - len(seq[:maxlen]))
        x_.append(padded_seq)




    y_ = []
    for seq in y:
        padded_seq = seq[:maxlen] + [0] * (maxlen - len(seq[:maxlen]))
        y_.append(padded_seq)



    x_train,x_test=split_list(lst=x_,per=per)
    y_train,y_test=split_list(lst=y_,per=per)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    vocab_size = len(vocab_list)
    embed_dim = embed_dim
    num_heads = num_heads
    ff_dim = ff_dim
    num_blocks = num_blocks
    dropout_rate = dropout_rate
    num_encoders = num_encoders
    num_decoders = num_decoders

    model = build_transformer_model(maxlen, vocab_size, embed_dim, num_heads, ff_dim, num_blocks, dropout_rate,
                                    num_encoders, num_decoders)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit([x_train, x_train], y_train, validation_data=([x_test, x_test], y_test), batch_size=batch_size, epochs=epochs)

    predictions = model.predict([x_test, x_test])
    predicted_classes = np.argmax(predictions, axis=-1)
    p=predicted_classes.tolist()
    for i in range(len(p)):
        output = number_to_words(lst=p[i], dictionary=d)
        print(output)
    return model,d,maxlen
def query_gen_sentences(query, model, dictionary, maxlen):
    query_order = return_order(query=query,dictionary=dictionary)
    u_order = np.array(query_order)
    padding_length = max(0, maxlen - len(u_order))
    padded_u_order = np.pad(u_order, (0, padding_length), mode='constant', constant_values=0)
    padded_u_order = np.reshape(padded_u_order, (1, -1))
    predictions = model.predict([padded_u_order, padded_u_order])
    predicted_classes = np.argmax(predictions, axis=-1)
    predicted_classes=predicted_classes.tolist()

    words = number_to_words(lst=predicted_classes[0], dictionary=dictionary)



    return words







'''

model, d, maxlen = train_model(dictionary_train=dictionary, maxlen=100, per=0.90,
                                   embed_dim=256, num_heads=32, ff_dim=128,
                                   num_blocks=8, dropout_rate=0.1, num_encoders=2, num_decoders=2, batch_size=64,
                                   epochs=100)

s=input("enter: ")
w=query_gen_sentences(query=s,model=model,dictionary=d,maxlen=maxlen)
print(w)

'''

