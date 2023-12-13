# GPT-neo

GPT-neo is a transformer model built from the ground up, inspired by the research paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). This model is designed for various natural language processing tasks, and for fun, I used it to create a clone of my favorite writer, Shakespeare.

## Table of Contents
- [Comparison with other llm](#comparison-with-other-llm)
- [How to Use](#how-to-use)
  - [Install Dependencies](#install-dependencies)
  - [Train](#train)
  - [Generate](#generate)
- [Working of GPT-neo](#working-of-gpt-neo)
  - [Training Process](#training-process)
  - [Generation Process](#generation-process)
- [Acknowledgements](#acknowledgements)

## Comparison with other llm


| **Configuration**        | **Your Model**          | **GPT-3**               | **GPT-3 Small**         |
|--------------------------|-------------------------|-------------------------|-------------------------|
| **Parameters**           | 26M                     | 175B                    | 125M                    |
| **Batch Size**           | 512                     | 3.2M                    | 0.5M                    |
| **Block Size**           | 50                      | -                       | -                       |
| **Maximum Iterations**   | 10,000                  | -                       | -                       |
| **Evaluation Interval**  | 100                     | -                       | -                       |
| **Learning Rate**        | 1e-3                    | 0.6 * 10^-4             | 6.0 * 10^-4             |
| **Device**               | 'cuda' if available     | -                       | -                       |
| **Evaluation Iterations**| 200                     | -                       | -                       |
| **Embedding Dimension**  | 64                      | 12,288                  | 768                     |
| **Number of Heads**      | 4                       | 96                      | 12                      |
| **Number of Layers**     | 5                       | 96                      | 12                      |
| **Dropout Rate**         | 0.0                     | -                       | -                       |
| **Token**                | 300K                    | 300B                    | 300B                    |


## How to Use

### Using Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Vf8llbJoM_ZgbXFw-21pS2PTbYdFbC1n#scrollTo=bIpNCuaoBciV)

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Train and Generate Text

To train the Bigram Language Model and generate text, use the following command:

```bash
python3 biagram.py
```

> **Note:** You have to pass arguments of `-e` or `-s` to change the training type from speed to efficiency.

> Example for efficiency training:
> ```bash
> python3 biagram.py -e
> ```

Feel free to explore and experiment with this model to generate creative and interesting text!

## Working of GPT-neo

### Training Process

1. **Data Preparation:** Gather a dataset suitable for the task at hand. In this case, you may use a collection of Shakespeare's works.

2. **Tokenization:** Break down the text into smaller units called tokens. Tokens can be words, subwords, or characters, depending on the chosen tokenization scheme.

3. **Model Architecture:** GPT-neo is based on the transformer architecture. It consists of an encoder-decoder structure with self-attention mechanisms. The model has multiple layers, each containing attention heads for capturing different aspects of the input data.

4. **Initialization:** Initialize the model with random weights. The weights will be adjusted during training to minimize the difference between the predicted and actual output.

5. **Training Loop:**
   - **Forward Pass:** Feed input tokens into the model to get predictions.
   - **Loss Computation:** Compare the model's predictions with the actual target tokens using a loss function (e.g., cross-entropy loss).
   - **Backward Pass:** Use backpropagation to calculate gradients of the loss with respect to the model parameters.
   - **Parameter Update:** Update the model parameters using optimization algorithms like Adam or SGD.

6. **Epochs:** Repeat the training loop for multiple epochs, iterating over the entire dataset each time. This allows the model to learn patterns and dependencies in the data.

### Generation Process

1. **Initialization:** Start with an initial input sequence or prompt. This could be a seed sentence or a few words.

2. **Tokenization:** Tokenize the input sequence to convert it into the format the model understands.

3. **Model Inference:**
   - **Forward Pass:** Feed the tokenized input into the trained model.
   - **Sampling:** Sample the next token from the predicted probability distribution. This can be done deterministically by choosing the token with the highest probability or stochastically by sampling from the distribution.

4. **Sequence Expansion:** Append the sampled token to the input sequence and repeat the process to generate the desired length of text.

5. **Output:** The generated sequence is the model's creative output based on the given input and its learned knowledge from the training data.

Remember that the quality of the generated text depends on the training data, model architecture, and hyperparameters chosen during training. Experimenting with these aspects can lead to more diverse and interesting results.

## Acknowledgements

All GPT-neo experiments are powered by GPUs on [Google Colab](https://colab.research.google.com/) and [Intel GPU Notebook](https://console.cloud.intel.com/), my favorite Cloud GPU provider.
Thanks to authors of the following papers for their research and contributions:
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-3](https://arxiv.org/abs/2005.14165)
