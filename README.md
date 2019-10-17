# Intelligent sentence generation for dialogue systems    

## Background of this project
Virtual personal assistants (VPAs) implemented using Deep Learning has the potential to provide assistance and finish tasks efficiently, such as Microsoft’s Cortana, Apple’s Siri, Amazon Alexa, Google Assistant and Facebook’s M. The advantage of neural models in dialogue modeling includes the handling of large scale data and varieties in data source. However, applying deep learning models to build robust and intelligent dialogue systems is challenging because it requires a deep understanding in natural language processing, prior work as well as the recent state-of-the-art work. Dialogue systems has been developed from simple sequence-to-sequence learning to Reinforcement Learning (RL) based sentence generation (Li et al. 2016) to Generative Adversarial Net (GAN) based sequence generation (SeqGAN) (Yu et al. 2016; Li et al. 2017). In the sequence-to-sequence learning model, Decoder system utilized LSTM-Recurrent neural network to analyze the input sentences and the sentence dependency while the Encoder works as generation system to output sentence responses. The training is optimized by minimizing cross entropy of each component. In RL for sentence generation, human responses are included to maximize expected reward through policy gradient. This implementation has improvement in ease of answering, information flow and semantic coherence and potential extension in any advanced RL model. In GAN implemented models, the generation and discrimination processes empowers Reinforcement Learning to update reward in every generation step. In the real data experiments, SeqGAN has improved in BLEU compared to MLE models. In this project, we would like to develop a chatbot using the approaches including seq2seq, RL and GAN. 


## Experimental datasets

Twitter dialogue corpora at Microsoft
This twitter corpora stored at Microsoft includes a collection of 12696 Tweet Ids representing 4,232 three-step conversational snippets extracted from Twitter logs.

Cornell Movie Dialogs Corpus (http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

This corpus consists a large number of fictional conversions exacted from 617 movies involving 9035 characters. In total, there are 220579 conversational exchanges between 10292 pairs of movie characters

Evaluations
BLEU	BLEU(bilingual evaluation understudy) measures word overlap to evaluate the quality of texts. 
Diversity	we will compare the diversity of generated responses of distinct unigrams and bigrams
Human Evaluation	we will include human evaluation in our generated responses.		

## Proposed approaches

Recurrent Neural Network and Language model 
Language model is necessary and useful in machine translation and word choice, as it takes into consideration of word contexts and calculates the probability of a sentence. In Recurrent Neural Network, a probability distribution over the vocabulary is predicted on the input word vectors and it is optimized using cross entropy in each components. 

Generative Adversarial Networks

Reinforcement Learning

Optimization


## Expected results

Ideally, we will generate results from the training processes to a simulated dialogue. 
In the training, we will present the hyperparameter selection and loss curves in the system optimization. Based on the evaluation metrics, we will report the performance between our dialogue system and the baseline seq2seq model. 


## References
Li, Jiwei, Will Monroe, Alan Ritter, Michel Galley, Jianfeng Gao, and Dan Jurafsky. 2016. “Deep Reinforcement Learning for Dialogue Generation.” arXiv [cs.CL]. arXiv. http://arxiv.org/abs/1606.01541.
Li, Jiwei, Will Monroe, Tianlin Shi, Sėbastien Jean, Alan Ritter, and Dan Jurafsky. 2017. “Adversarial Learning for Neural Dialogue Generation.” In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, 2157–69. Stroudsburg, PA, USA: Association for Computational Linguistics.
Yu, Lantao, Weinan Zhang, Jun Wang, and Yong Yu. 2016. “SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient.” arXiv [cs.LG]. arXiv. http://arxiv.org/abs/1609.05473.

