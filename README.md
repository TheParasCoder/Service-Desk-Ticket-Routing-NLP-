# Service-Desk-Ticket-Routing-NLP-
Service Desk Ticket Routing (NLP) 
# Service Desk Ticket Routing AI

An NLP-based system to classify IT support tickets using a fine-tuned DistilBERT model. This project demonstrates a full MLOps pipeline including synthetic data generation, model training, data integrity checks (Deepchecks), and automated deployment logic.

## Project Structure

- `src/data_gen.py`: Generates synthetic IT support tickets (Hardware, Billing, Access).
- `src/train.py`: Fine-tunes a DistilBERT model on the generated data.
- `src/evaluate.py`: Runs Deepchecks for data integrity and evaluates the model against a baseline (Logic Gate).
- `.github/workflows/deploy.yml`: CI/CD pipeline definition.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate Dataset:
   ```bash
   python src/data_gen.py
   ```

3. Train Model:
   ```bash
   python src/train.py
   ```
   *Note: Training will take some time due to transformer fine-tuning.*

4. Evaluate & Verify:
   ```bash
   python src/evaluate.py
   ```

## MLOps Features

- **Deepchecks Integration**: Before evaluation, the data is checked for integrity issues.
- **Logic Gate**: The new model is only "deployed" if its Weighted F1-Score improves upon the simulated production baseline (0.85) by at least 2%.
