# Tranformers

-------------

## Transformers Basics

![Tranformers](https://lh5.googleusercontent.com/oMRBmT0Do5V4mSh_oYMq7HgTqEIt7tz6rhs_oLdfwWIY1pFUcjj95P7AU7SKdoM8ZOTLlqw_J2nAj_iCsqUi00KXJAmHZcf_h2c1ul-zLf4BgERRU6hVBa3IMeceP9juFNhBz2FybwDF6_714msbDUAk4pBTgcxUpax_qE5Ar9b9o0LWbKJA-jmqVtR5)

- Fundamentally, the transformer is a **sequence modelling model**. The input is a sequence of *tokens* (roughly, sub-words) which are mapped to the output, a tensor of logits, which are mapped by a softmax to a probability distribution over possible next tokens.
  - We call the input sequence of tokens the context
- A transformer consists of an embedding layer, followed by n transformer blocks/transformer layers, and finally a linear unembed layer which maps the model’s activations to the output logits.
  - Confusingly, a transformer layer actually contains two layers, an attention layer and an MLP layer.
- Internally, the central object of a transformer is the **residual stream.**The residual stream after layer n is the sum of the embedding and the outputs of all layers up to layer n, and is the input to layer n+1.
  - In the standard framing of a neural network, we think of the output of layer n being fed into the input of layer n+1. The residual stream can fit into this framing by thinking of it as a series of **skip connections** - an identity map around the layer, so output_n = output_layer_n + skip_connection_n = output_layer_n + input_layer_n.
    - I think this is a less helpful way to think about things though, as in practice the skip connection conveys far more information than the output of any individual layer, and information output by layer n is often only used by layers n+2 and beyond.
  - The residual stream can be thought of as a **shared bandwidth/shared memory**of the transformer - it’s the only way that information can move between layers, and so the model needs to find a way to store all relevant information in there. 
    - We might expect to find different subspaces dedicated to different kinds of information - eg "the current token is", "the subject of the sentence is", "the previous token is", "the position of the start of the line is", etc
  - **Memory management** is the phenomena when part of the model are used to manage this shared memory. Eg if one head/neuron's output is used by a certain layer and then no longer needed, a later layer neuron may delete it. 
    - One motif that can indicate this is a neuron with significant negative cosine similarity between its input and output vector
