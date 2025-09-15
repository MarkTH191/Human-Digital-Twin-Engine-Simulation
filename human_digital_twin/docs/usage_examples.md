# Usage Examples

## Run the GUI

```bash
streamlit run main.py
```

## Example Workflow
- Open the app and load the default mini knowledge graph.
- Select an organ (e.g., Liver), pick a tissue and cell type.
- Apply an intervention: knock down a gene or add a drug.
- Accelerate time to observe outcomes.
- Export results as JSON.

## API Server

```bash
python -m api.server
```

Then open `http://localhost:8000/docs`.