---
license: cc-by-nc-nd-4.0
language:
- en
base_model:
- naver/trecdl22-crossencoder-debertav3
pipeline_tag: text-ranking
---

# Model Card for Provence-reranker

<img src="https://cdn-uploads.huggingface.co/production/uploads/6273df31c3b822dad2d1eef2/4n-bxYfiMPC2LoLM2m7pg.png" alt="image/png" width="600">

Provence is a lightweight **context pruning model** for retrieval-augmented generation, particularly **optimized for question answering**. Given a user question and a retrieved passage, Provence **removes sentences from the passage that are not relevant to the user question**. This **speeds up generation** and **reduces context noise**, in a plug-and-play manner **for any LLM**. 

*Paper*: https://arxiv.org/abs/2501.16214, accepted to ICLR 2025

*Blogpost*: https://huggingface.co/blog/nadiinchi/provence

*Developed by*: Naver Labs Europe  
*License*: Provence is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license [CC BY-NC-ND 4.0 license]. [License file](https://huggingface.co/naver/provence-reranker-debertav3-v1/blob/main/Provence_LICENSE.txt)
* *Model*: `provence-reranker-debertav3-v1` (Provence for Pruning and Reranking Of retrieVEd relevaNt ContExt)  
* *Backbone model*: [DeBERTav3-reranker](https://huggingface.co/naver/trecdl22-crossencoder-debertav3) (trained from [DeBERTa-v3-large](https://huggingface.co/microsoft/deberta-v3-large))  
* *Model size*: 430 million parameters  
* *Context length*: 512 tokens

_NEW!_ The multilingual version of the model, based on BGE-reranker-v2-m3, is available [here](https://huggingface.co/naver/xprovence-reranker-bgem3-v1).

Training and evaluation code & data are available in the [Bergen repo](https://github.com/naver/bergen/tree/main/scripts/provence)

## Usage

Provence uses `nltk`:

```bash
pip install nltk
python -c "import nltk; nltk.download('punkt_tab')"
```

Pruning a single context for a single question:

```python
from transformers import AutoModel

provence = AutoModel.from_pretrained("naver/provence-reranker-debertav3-v1", trust_remote_code=True)

context = "Shepherd’s pie. History. In early cookery books, the dish was a means of using leftover roasted meat of any kind, and the pie dish was lined on the sides and bottom with mashed potato, as well as having a mashed potato crust on top. Variations and similar dishes. Other potato-topped pies include: The modern ”Cumberland pie” is a version with either beef or lamb and a layer of bread- crumbs and cheese on top. In medieval times, and modern-day Cumbria, the pastry crust had a filling of meat with fruits and spices.. In Quebec, a varia- tion on the cottage pie is called ”Paˆte ́ chinois”. It is made with ground beef on the bottom layer, canned corn in the middle, and mashed potato on top.. The ”shepherdess pie” is a vegetarian version made without meat, or a vegan version made without meat and dairy.. In the Netherlands, a very similar dish called ”philosopher’s stew” () often adds ingredients like beans, apples, prunes, or apple sauce.. In Brazil, a dish called in refers to the fact that a manioc puree hides a layer of sun-dried meat."
question = 'What goes on the bottom of Shepherd’s pie?'

provence_output = provence.process(question, context)
# print(f"Provence Output: {provence_output}")
# Provence Output: {'reranking_score': 3.022725, pruned_context': 'In early cookery books, the dish was a means of using leftover roasted meat of any kind, and the pie dish was lined on the sides and bottom with mashed potato, as well as having a mashed potato crust on top.']]
```

You can also pass a list of questions and a list of lists of contexts (multiple contexts per question to be pruned) for batched processing.

Setting `always_select_title=True` will keep the first sentence "Shepherd’s pie". This is especially useful for Wikipedia articles where the title is often needed to understand the context.
More details on how the title is defined are given below.

```python
provence_output = provence.process(question, context, always_select_title=True)
# print(f"Provence Output: {provence_output}")
# Provence Output: {'reranking_score': 3.022725, pruned_context': 'Shepherd’s pie. In early cookery books, the dish was a means of using leftover roasted meat of any kind, and the pie dish was lined on the sides and bottom with mashed potato, as well as having a mashed potato crust on top.']]
```

## Model interface

Interface of the `process` function:
* `question`: `Union[List[str], str]`: an input question (str) or a list of input questions (for batched processing)
* `context`: `Union[List[List[str]], str]`: context(s) to be pruned. This can be either a single string (in case of a singe str question), or a list of lists contexts (a list of contexts per question), with `len(contexts)` equal to `len(questions)`
*  `title`: `Optional[Union[List[List[str]], str]]`, _default: “first_sentence”_: an optional argument for defining titles. If `title=first_sentence`, then the first sentence of each context is assumed to be the title. If `title=None`, then it is assumed that no titles are provided. Titles can be also passed as a list of lists of str, i.e. titles shaped the same way as contexts. Titles are only used if `always_select_title=True`.
* `threshold` _(float, $ \in [0, 1]$, default 0.1)_: which threshold to use for context pruning. We recommend 0.1 for more conservative pruning (no performance drop or lowest performance drops) and 0.5 for higher compression, but this value can be further tuned to meet the specific use case requirements.
* `always_select_title` _(bool, default: True)_: if True, the first sentence (title) will be included into the selection each time the model select a non-empty selection of sentences. This is important, e.g., for Wikipedia passages, to provide proper contextualization for the next sentences.
* `batch_size` (int, default: 32)
* `reorder` _(bool, default: False)_: if True, the provided contexts for each question will be reordered according to the computed question-passage relevance scores. If False, the original user-provided order of contexts will be preserved.
* `top_k` _(int, default: 5)_: if `reorder=True`, specifies the number of top-ranked passages to keep for each question.
* `enable_warnings` _(bool, default: True)_: whether the user prefers the warning about model usage to be printed, e.g. too long contexts or questions.

## Model features

* **Provence encodes all sentences in the passage together**: this enables capturing of coreferences between sentences and provides more accurate context pruning.
* **Provence automatically detects the number of sentences to keep**, based on a threshold. We found that the default value of a threshold works well across various domains, but the threshold can be adjusted further to better meet the particular use case needs.
* **Provence is robust to various domains**, being trained on a combination of diverse MS Marco and Natural Questions data.
* **Provence works out-of-the-box with any LLM**.


## Model Details

* Input: user question (e.g., a sentence) + retrieved context passage (e.g., a paragraph)
* Output: pruned context passage, i.e., irrelevant sentences are removed + relevance score (can be used for reranking)
* Model Architecture: The model was initialized from [DeBERTav3-reranker](https://huggingface.co/naver/trecdl22-crossencoder-debertav3) and finetuned with two objectives: (1) output a binary mask which can be used to prune irrelevant sentences; and (2) preserve initial reranking capabilities. 
* Training data: MS Marco (document) + NQ training sets, with synthetic silver labelling of which sentences to keep, produced using [LLama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B).
* Languages covered: English
* Context length: 512 tokens (similar to the pretrained DeBERTa model)
* Evaluation: we evaluate Provence on 7 datasets from various domains: Wikipedia, biomedical data, course syllabi, and news. Evaluation is conducted on the model trained only on MS Marco data. We find that Provence is able to prune irrelevant sentences with little-to-no drop in performance, in all domains, and outperforms existing baselines on the Pareto front (top right corners of the plots).

Check out more analysis in the [paper](https://arxiv.org/abs/2501.16214)!

<img src="https://cdn-uploads.huggingface.co/production/uploads/6273df31c3b822dad2d1eef2/WMmfsNG48O830paaBAaQF.png" width="600">


## License

This work is licensed under CC BY-NC 4.0. 

## Cite

```
@misc{chirkova2025provenceefficientrobustcontext,
      title={Provence: efficient and robust context pruning for retrieval-augmented generation}, 
      author={Nadezhda Chirkova and Thibault Formal and Vassilina Nikoulina and Stéphane Clinchant},
      year={2025},
      eprint={2501.16214},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.16214}, 
}
```

## Acknowledgements

Model trained at [Naver Labs Europe](https://europe.naverlabs.com/)  
Team:
* [Nadia Chirkova](https://nadiinchi.github.io/)
* [Thibault Formal](https://europe.naverlabs.com/people_user_naverlabs/thibault-formal/)
* [Vassilina Nikoulina](https://europe.naverlabs.com/people_user_naverlabs/vassilina-nikoulina/)
* [Stéphane Clinchant](https://europe.naverlabs.com/people_user_naverlabs/st%C3%A9phane-clinchant/)