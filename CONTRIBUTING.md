# Contributing to MIND

Thank you for your interest in contributing to **MIND: MRI-based Intelligent Neural Detection**.

MIND is an educational and research project for brain MRI image classification using deep learning. Contributions are welcome, especially improvements to documentation, code quality, model training, inference, evaluation, and explainability.

## Medical Disclaimer

This project is not intended for clinical diagnosis or medical decision-making. Any contribution that changes model behavior, prediction output, or medical wording should preserve this limitation clearly.

## Ways to Contribute

You can contribute by:

- Improving documentation and setup instructions
- Fixing bugs in the Streamlit app, training script, or inference pipeline
- Improving code formatting, linting, and project structure
- Adding tests or validation scripts
- Improving model evaluation and reporting
- Adding explainability features such as Grad-CAM
- Improving UI/UX for the Streamlit demo
- Reporting issues or unclear behavior

## Getting Started

Fork the repository and clone your fork:

\`\`\`bash
git clone https://github.com/YOUR_USERNAME/mind-mri-brain-tumor.git
cd mind-mri-brain-tumor
\`\`\`

Create and activate a virtual environment:

\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
\`\`\`

Install dependencies:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

Run the Streamlit app:

\`\`\`bash
streamlit run app.py
\`\`\`

## Development Guidelines

Before submitting a pull request, please make sure that:

- Your code is clear and readable
- Your changes do not remove the medical disclaimer
- Your changes do not commit private data or large dataset files
- The app still runs locally with \`streamlit run app.py\`
- Any model-related change is documented clearly
- New dependencies are added to \`requirements.txt\`

## Dataset Guidelines

The training dataset is not included in this repository. Contributors should not upload the Kaggle dataset or other medical image datasets directly to the repository.

If your contribution depends on data, document:

- Dataset source
- Expected folder structure
- Preprocessing steps
- Any assumptions about image format or class labels

## Model and Evaluation Changes

For changes related to model training or evaluation, please include:

- Model architecture or configuration changes
- Training settings such as epochs, batch size, optimizer, and learning rate
- Dataset split details
- Evaluation metrics
- Any known limitations

Avoid claiming clinical performance unless the model has been validated in an appropriate clinical setting.

## Commit Style

Use short and descriptive commit messages, for example:

\`\`\`bash
git commit -m "Fix image preprocessing in inference"
git commit -m "Improve README setup instructions"
git commit -m "Add Grad-CAM utility placeholder"
\`\`\`

## Pull Request Checklist

Before opening a pull request, check that:

- [ ] The app runs locally
- [ ] Code changes are documented
- [ ] New dependencies are listed in \`requirements.txt\`
- [ ] No dataset files, private files, or large temporary files are committed
- [ ] Medical disclaimer is preserved
- [ ] The change is explained clearly in the pull request description

## Reporting Issues

When reporting a bug, please include:

- Operating system
- Python version
- Steps to reproduce the issue
- Error message or screenshot
- Expected behavior
- Actual behavior

For model prediction issues, include the image source only if you have permission to share it.

## License

By contributing to this project, you agree that your contributions will be licensed under the repository's MIT License.
EOF
