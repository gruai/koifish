## Weight Tying(WT)

### Advantages & disadvantages
1. WT would reduces the memory,  may results in better and faster outcomes for small language model.
2. According to findings from OLMo the weight tying is beneficial for smaller models like 1B but for larger ones starting from 7B it starts to hurt the performance - instability in loss curves. One researcher is talking about it in TWIML AI podcast around 16:50: https://youtu.be/mwS9zPCv_dY?t=1010
3. Scaling - Output softmax wants embeddings to be very large so their inner products will produce very different values.
Input embeddings want a much smaller range so they can have stable dynamics throughout training. All the "old" code bases had this scalar (usually sqrt(d)) but the llama arch dropped this.
4. Use Weight Tying only when the Distributional Hypothesis Holds.

### Reference
1. [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
2. https://www.reddit.com/r/MachineLearning/comments/1d2iurw/d_should_the_embedding_matrix_and_final/

### Practice of Koifish
Since Koifish can only train small and medium-sized models at present, it always adopts Weight Tying(WT) technology.
There is no scaling before output sofmax layer. But we may try this in the future.