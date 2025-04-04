<h1 align="center">📊 RefChartQA: Grounding Visual Answer on Chart Images</h1>
<p align="center">
  <a href="https://www.linkedin.com/in/alevogel/"><strong>Alexander Vogel</strong></a>
  ·
  <a href="https://www.linkedin.com/in/omar-moured/"><strong>Omar Moured</strong></a>
  ·
  <a href="https://scholar.google.com/citations?user=nG2ebe8AAAAJ&hl=en"><strong>Yufan Chen</strong></a>
  ·
  <a href="https://jamycheung.github.io/"><strong>Jiaming Zhang</strong></a>
  ·
  <a href="https://cvhci.iar.kit.edu/"><strong>Rainer Stiefelhagen</strong></a>
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/omoured/RefChartQA">
    <img src="https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface" alt="Hugging Face">
  </a>
  &nbsp;
  <a href="https://arxiv.org/abs/2503.23131">
    <img src="https://img.shields.io/badge/Paper-ArXiv-blue?logo=arxiv" alt="Paper">
  </a>
</p>

<p align="center">
  <img src="teaser.png" alt="Teaser image" width="80%">
</p>


## 🧠 About

**RefChartQA** is a large-scale benchmark for **visual grounding in chart-based question answering**. It extends the ChartQA and TinyChart-PoT datasets by adding explicit bounding box annotations that link each answer to supporting visual elements in the chart. RefChartQA contains **73,702 annotated samples**, including:
- **55,789** training,
- **6,223** validation,
- **11,690** testing instances.

<p align="center">
  <img src="datasetsamples.png" alt="Teaser image" width="80%">
</p>

## ✅ To-Do

- [x] [March 28, 2025] Upload dataset to Hugging Face 🤗  
- [ ] Release evaluation scripts.
- [ ] Add visualization tools for bounding box inspection.

## 🛠️ Usage

### 📦 Environment Setup

Make sure you have Python installed, then install the required libraries:

```python
pip install datasets pillow
```

### 📂 Load the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("omoured/RefChartQA")

# Print dataset splits and one example
print(dataset)
print(dataset["train"][0])
```

## 📚 Citation
```bibtex
@misc{vogel2025refchartqagroundingvisualanswer,
      title={RefChartQA: Grounding Visual Answer on Chart Images through Instruction Tuning}, 
      author={Alexander Vogel and Omar Moured and Yufan Chen and Jiaming Zhang and Rainer Stiefelhagen},
      year={2025},
      eprint={2503.23131},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.23131}, 
}
```