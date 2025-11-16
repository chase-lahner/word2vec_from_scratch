# word2vec_from_scratch

I decided to make this in a weekend in order to get a deeper understanding of Word2Vec and creating a neural network from scratch. This implementation is purely inspired from this [blog post](https://jaketae.github.io/study/word2vec/) which is insanely good and I highly recommend. I plan to create more projects like this.

This implementation uses the skip-gram model, which essentially means you give a word as input, and the classification task is that the model tries to predict the context words around the target word, around a window size.

Word2Vec uses a simple single-hidden layer neural network, with a softmax on the output layer, and the cool thing is that the learned weight matrix (w1) is actually the embeddings! We use cross-entropy as the loss function, and a hidden layer of 10 neurons.
