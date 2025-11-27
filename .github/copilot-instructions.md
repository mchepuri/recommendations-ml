# Copilot Instructions for Recommendations-ML

## Project Overview

This is a UC Berkeley machine learning project focused on building recommendation systems. The project uses Jupyter notebooks as the primary development environment for exploratory data analysis, model development, and evaluation.

**Key Files:**
- `recommendations-ml.ipynb` - Main notebook containing analysis, modeling, and results
- `README.md` - Project documentation (template-based structure from UC Berkeley)

## Development Patterns

### Notebook-Driven Development

The project follows a Jupyter notebook-centric workflow:
- All analysis, experiments, and model development occur in `recommendations-ml.ipynb`
- Use markdown cells to structure the notebook with clear section headers
- Follow the UC Berkeley template structure outlined in `README.md`:
  1. Executive summary
  2. Rationale (why this matters)
  3. Research question
  4. Data sources
  5. Methodology
  6. Results
  7. Next steps

**Example workflow:**
- Add data exploration cells early to understand the dataset
- Document methodology cells with explanations before implementing
- Include visualization cells after analysis sections for clarity
- Place conclusion/next steps cells at the end

### Code Organization in Notebooks

When adding code cells:
- Group related operations (data loading, preprocessing, modeling, evaluation)
- Use markdown cells as section dividers with clear headings
- Import dependencies at the top of the notebook
- Add comments for complex transformations or model-specific logic

## Data Science Conventions

### Common Libraries Expected
- pandas for data manipulation
- numpy for numerical operations
- scikit-learn for ML algorithms
- matplotlib/seaborn for visualization
- Standard recommendation libraries (e.g., surprise, implicit, collab filtering)

### Results & Evaluation

Document results using:
- Numerical metrics appropriate to recommendation systems (RMSE, precision@k, recall@k, nDCG)
- Visualization charts (confusion matrices, ranking metrics, performance comparisons)
- Clear markdown explanations of findings and implications

## Project Structure

```
recommendations-ml/
├── recommendations-ml.ipynb    # Main analysis notebook
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
└── .github/
    └── copilot-instructions.md # This file
```

## Key Workflows

### Running the Project

Execute the notebook top-to-bottom:
1. Data loading and exploration
2. Data preprocessing and feature engineering
3. Model training and hyperparameter tuning
4. Model evaluation and comparison
5. Results visualization and documentation

### Extending the Project

When adding new analysis:
1. Create new markdown section headers to organize content
2. Add data exploration cells before modeling
3. Document assumptions and methodological choices
4. Include visualization cells to validate results
5. Update the outline section in README.md with links to key sections

## Notes for AI Agents

- The project is template-driven and early-stage; expect to fill in the README structure as work progresses
- Focus on the Research Question and Methodology to guide your work
- Always include visualization and explanation cells alongside computational cells
- Maintain clear documentation in markdown to explain findings to non-technical stakeholders
- For recommendation systems specifically, consider evaluation metrics beyond accuracy (coverage, diversity, novelty)
