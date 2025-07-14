# Multi-Stage Domain Adaptation for Neural Machine Translation

A hybrid learning framework for specialized terminology and low-resource language pairs, demonstrated on Chinese religious text translation.

## Overview

This framework addresses domain adaptation challenges in neural machine translation through a multi-stage computational approach that combines:

- **Automated terminological extraction** for all language pairs
- **Neural fine-tuning** for mBART-supported modern languages  
- **Systematic terminology management** across diverse resource scenarios

## Key Features

- **Multi-stage processing pipeline**: Dictionary matching → Similarity analysis → Statistical dominance → Proper noun detection
- **Hybrid approach**: Pure terminological learning for ancient languages (Hebrew, Greek, Latin) + neural enhancement for modern languages
- **Scalable architecture**: Modular design adaptable to other specialized domains
- **Comprehensive evaluation**: Multiple metrics including BLEU, ROUGE-L, and COMET scores

## Performance Results

- **1.13x to 2.97x improvement** in terminological learning compared to baseline dictionary approaches
- **Consistent gains** in BLEU/ROUGE/COMET scores for neural-enhanced language pairs
- **Effective processing** of both resource-constrained ancient languages and modern languages

## Framework Architecture

```
Input Texts (XML) → Text Preparation → Core Processing → Outputs
                                           ↓
                              Learning Module (Universal)
                              Fine-tuning Module (mBART supported)
                                           ↓
                              Learned Dictionary + Fine-tuned Models
```

## Installation

```bash
git clone https://github.com/AlexisBal/specialized-nmt-framework.git
cd specialized-nmt-framework
pip install -r requirements.txt
```

## Quick Start

1. **Prepare your data**: Place source and target language XML files in the `input/` directory
2. **Configure settings**: Edit configuration files for your language pair
3. **Run the pipeline**:
   ```bash
   python main.py --source-lang [LANG] --target-lang zh --mode [terminological|hybrid]
   ```

## Supported Languages

### Terminological Learning (All language pairs)
- Ancient languages: Hebrew, Greek, Latin
- Modern languages: English, French, Spanish, Portuguese, German, Italian
- Target: Chinese

### Neural Fine-tuning (mBART supported)
- Source: English, French, Spanish, Portuguese, German, Italian
- Target: Chinese

## Directory Structure

```
├── data_preparation/     # Text preprocessing and alignment
├── learning_modules/     # Terminological extraction algorithms
├── tuner/               # Neural fine-tuning components
├── text_organizer/      # Intelligent text segmentation
├── memory_manager/      # Batch processing and caching
├── utils/               # Helper functions and tools
├── input/               # Input corpus files
├── outputs_template/    # Output directory template
└── main.py             # Main processing script
```

## Use Cases

- **Academic research**: Historical text analysis and cross-cultural terminology studies
- **Specialized translation**: Legal, medical, and technical documentation
- **Digital humanities**: Computational approaches to classical texts
- **Multilingual lexicography**: Automated dictionary construction

## Technical Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Sentence Transformers
- Additional dependencies in `requirements.txt`

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{balmont2024specialized,
  title={Multi-Stage Domain Adaptation for Neural Machine Translation: A Hybrid Learning Framework for Specialized Terminology and Low-Resource Language Pairs},
  author={Balmont, Alexis},
  journal={Digital Humanities Quarterly},
  year={2024},
  url={https://github.com/AlexisBal/specialized-nmt-framework}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Contact

- **Author**: Alexis Balmont
- **Email**: alexis.balmont@gmail.com
- **Affiliation**: Chinese University of Hong Kong
- **ORCID**: https://orcid.org/0009-0004-9415-1818

## Acknowledgments

This research demonstrates computational approaches to specialized domain translation while acknowledging the importance of human expertise in professional translation applications.
