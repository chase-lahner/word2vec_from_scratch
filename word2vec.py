import numpy as np
from nltk import word_tokenize
import string
import matplotlib.pyplot as plt

text = '''Machine learning is the study of computer algorithms that \
improve automatically through experience. It is seen as a \
subset of artificial intelligence. Machine learning algorithms \
build a mathematical model based on sample data, known as \
training data, in order to make predictions or decisions without \
being explicitly programmed to do so. Machine learning algorithms \
are used in a wide variety of applications, such as email filtering \
and computer vision, where it is difficult or infeasible to develop \
conventional algorithms to perform the needed tasks.'''


def tokenize(text):
    text = "".join([letter for letter in text if letter not in string.punctuation])
    return list(word_tokenize(text.lower()))


def create_lookup(tokenized):
    word_to_id = {}
    id_to_word = {}
    
    for i, token in enumerate(set(tokenized)):
        word_to_id[token] = i
        id_to_word[i] = token
    return word_to_id, id_to_word
        

def generate_train_examples(text, word_to_id, id_to_word, window_size):
    inputs = []
    predictions = []
    # given each token in the text, we need to:
    # 1) Look in window_size to the left and the right
    # 2) create entry in the form of [target_word, context_word] ex. [Machine, learning]
    #) Add [Machine] to X (input)
    # Add [Learning] to Y (output)
    # repeat until context window finished (in this case [Machine] [is] would be the only other pair)
    for i, token in enumerate(text):
        left_range = range(max(0, i-window_size), i)
        right_range = range(i, min(len(text), i + window_size + 1))
        combined_range = [*left_range, *right_range]
        for j in combined_range:
            if i == j:
                continue
            print("text", text[i])
            print(one_hot_encode(len(word_to_id), word_to_id[text[i]]))
            inputs.append(one_hot_encode(len(word_to_id), word_to_id[text[i]]))
            predictions.append(one_hot_encode(len(word_to_id), word_to_id[text[j]]))
            
        
    return np.array(inputs), np.array(predictions)


def one_hot_encode(len_vocab, word_index):  
    arr = np.zeros(len_vocab)
    arr[word_index] = 1
    return arr


def model(num_neurons, vocab_size):
    return {
        "w1": np.random.randn(vocab_size, num_neurons),
        "w2": np.random.randn(num_neurons, vocab_size)
    }
    
    
def forward(model, X):
    cache = {}
    cache["A1"] = X @ model["w1"]
    cache["A2"] = cache["A1"] @ model["w2"]
    cache["Z"] = softmax(cache["A2"])
    return cache
    
    
def softmax(vec):
    res = []
    for x in vec:
        es = np.exp(x)
        res.append(es / (np.sum(es)))
    return res

def backward( model, x, y, alpha):
    cache = forward(model, x)
    da2 = cache["Z"] - y
    dw2 = cache["A1"].T @ da2
    da1 = da2 @ model["w2"].T
    dw1 = x.T @ da1
    assert(dw1.shape == model["w1"].shape)
    assert(dw2.shape == model["w2"].shape)
    model["w1"]  -= alpha * dw1
    model["w2"] -= alpha * dw2
    return cross_entropy(cache["Z"], y)
    
    
def cross_entropy(z, y):
    return - np.sum(np.log(z) * y)

 
    
    




if __name__ == "__main__":
    tokenized = tokenize(text)
    word_to_id, id_to_word = create_lookup(tokenized)
    print(word_to_id)
    inputs, predictions = generate_train_examples(tokenized, word_to_id, id_to_word, 2)
    print(predictions.shape, inputs.shape)
    print(inputs[0:2])
    n_iter = 50
    learning_rate = 0.05
    t_model = model(10, len(word_to_id))
    history = [backward(t_model, inputs, predictions, learning_rate) for _ in range(n_iter)]
    
    plt.plot(range(len(history)), history, color="skyblue")
    plt.show()
    
    learning = one_hot_encode(len(word_to_id), word_to_id["learning"],)
    result = forward(t_model, [learning])["Z"][0]
    
    for word in (id_to_word[id] for id in np.argsort(result)[::-1]):
        print(word)

   
    
    

  