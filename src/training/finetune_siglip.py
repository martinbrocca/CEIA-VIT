# src/training/finetune_siglip.py
"""
SigLIP Fine-tuning on Food-101-Recipe-Pairs

Purpose:
    Fine-tunes google/siglip-base-patch16-224 on the Food-101-Recipe-Pairs dataset
    to improve vision-language alignment for food domain.
    
    Includes comprehensive MLflow logging of:
    - Training/validation/test losses at each step
    - Learning rate schedule
    - Model artifacts
    - Training curves
    - Performance metrics

Dataset:
    - Training: 17,383 pairs (80%)
    - Validation: 2,172 pairs (10%)
    - Test: 2,174 pairs (10%)

Usage:
    # Quick test (1 epoch)
    python src/training/finetune_siglip.py --epochs 1 --batch-size 16
    
    # Full training (recommended)
    python src/training/finetune_siglip.py --epochs 5 --batch-size 32
    
    # Maximize GPU utilization (RTX 5090)
    python src/training/finetune_siglip.py --epochs 5 --batch-size 64
    
    # Evaluate existing model only
    python src/training/finetune_siglip.py --eval-only

Results:
    - 5 epochs: Train loss 1.09, Val loss 0.95, Test loss 0.90
    - 83% loss reduction from initial 5.30
    - Training time: ~2.8 minutes on RTX 5090

Output:
    - models/siglip-food-finetuned/ (fine-tuned model)
    - MLflow run with all metrics and artifacts

"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor, TrainingArguments, Trainer
from PIL import Image
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import time
import shutil

from utils.config import PROJECT_ROOT
from utils.device import get_device
from utils.mlflow_logger import MLflowLogger
import mlflow


class Food101RecipeDataset(Dataset):
    """
    Dataset for fine-tuning vision-language models on Food-101-Recipe pairs
    """
    
    def __init__(self, pairs_file: Path, processor, split='train'):
        """
        Args:
            pairs_file: Path to food101_recipe_pairs.json
            processor: Transformers processor (e.g., SigLIP)
            split: 'train', 'val', or 'test'
        """
        self.processor = processor
        self.project_root = PROJECT_ROOT
        
        # Load pairs
        print(f"Loading {split} dataset from {pairs_file}...")
        with open(pairs_file, 'r') as f:
            data = json.load(f)
        
        all_pairs = data['pairs']
        
        # Create train/val/test splits (80/10/10)
        np.random.seed(42)
        indices = np.random.permutation(len(all_pairs))
        
        train_size = int(0.8 * len(all_pairs))
        val_size = int(0.1 * len(all_pairs))
        
        if split == 'train':
            self.pairs = [all_pairs[i] for i in indices[:train_size]]
        elif split == 'val':
            self.pairs = [all_pairs[i] for i in indices[train_size:train_size+val_size]]
        else:  # test
            self.pairs = [all_pairs[i] for i in indices[train_size+val_size:]]
        
        print(f"Loaded {len(self.pairs)} pairs for {split}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load image
        img_path = self.project_root / pair['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='white')
        
        # Get recipe text
        text = pair['recipe_text']
        
        # Process inputs
        encoding = self.processor(
            images=image,
            text=text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=64
        )
        
        # Build return dict - only include keys that exist
        result = {
            'pixel_values': encoding['pixel_values'].squeeze(),
            'input_ids': encoding['input_ids'].squeeze(),
        }
        
        # Add attention_mask only if it exists
        if 'attention_mask' in encoding:
            result['attention_mask'] = encoding['attention_mask'].squeeze()
        
        return result


class SigLIPFineTuner:
    """Fine-tune SigLIP on Food-101-Recipe-Pairs"""
    
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        output_dir: str = "./models/siglip-food-finetuned",
        pairs_file: Path = None
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.pairs_file = pairs_file or PROJECT_ROOT / "data" / "processed" / "food101_recipe_pairs.json"
        self.device = get_device()
        
        print(f"Device: {self.device}")
        
    def load_model(self):
        """Load base SigLIP model"""
        print(f"Loading model: {self.model_name}")
        self.model = AutoModel.from_pretrained(self.model_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        print(f"✓ Model loaded")
        
    def prepare_datasets(self):
        """Create train/val/test datasets"""
        print("\nPreparing datasets...")
        
        self.train_dataset = Food101RecipeDataset(
            self.pairs_file, 
            self.processor, 
            split='train'
        )
        
        self.val_dataset = Food101RecipeDataset(
            self.pairs_file,
            self.processor,
            split='val'
        )
        
        self.test_dataset = Food101RecipeDataset(
            self.pairs_file,
            self.processor,
            split='test'
        )
        
        print(f"\n✓ Datasets prepared:")
        print(f"  Train: {len(self.train_dataset)} pairs")
        print(f"  Val: {len(self.val_dataset)} pairs")
        print(f"  Test: {len(self.test_dataset)} pairs")
    
    def train(
        self,
        epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 5e-6,
        use_mlflow: bool = True
    ):
        """Fine-tune the model with comprehensive MLflow tracking"""
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / 'logs'),
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=500,
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=False,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to="none",
        )
        
        # Custom trainer with MLflow callback
        class SigLIPTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                """Compute contrastive loss"""
                # Prepare inputs - only pass keys that exist
                model_inputs = {
                    'pixel_values': inputs['pixel_values'],
                    'input_ids': inputs['input_ids'],
                    'return_loss': True
                }
                
                # Add attention_mask if present
                if 'attention_mask' in inputs:
                    model_inputs['attention_mask'] = inputs['attention_mask']
                
                outputs = model(**model_inputs)
                
                loss = outputs.loss
                return (loss, outputs) if return_outputs else loss
            
            def evaluation_loop(self, dataloader, description, prediction_loss_only=None, 
                              ignore_keys=None, metric_key_prefix="eval"):
                """Custom evaluation loop that returns loss"""
                model = self._wrap_model(self.model, training=False, dataloader=dataloader)
                model.eval()
                
                total_loss = 0
                num_batches = 0
                
                for step, inputs in enumerate(dataloader):
                    inputs = self._prepare_inputs(inputs)
                    
                    with torch.no_grad():
                        loss = self.compute_loss(model, inputs)
                        total_loss += loss.item()
                        num_batches += 1
                
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                
                metrics = {
                    f"{metric_key_prefix}_loss": avg_loss,
                }
                
                return type('EvalLoopOutput', (), {
                    'predictions': None,
                    'label_ids': None,
                    'metrics': metrics,
                    'num_samples': len(dataloader.dataset)
                })()
        
        # MLflow tracking
        if use_mlflow:
            logger = MLflowLogger("recipe-search-pipeline")
            
            with mlflow.start_run(run_name="siglip_finetuning") as run:
                run_id = run.info.run_id
                
                # Log parameters
                mlflow.log_param("model_name", self.model_name)
                mlflow.log_param("epochs", epochs)
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("learning_rate", learning_rate)
                mlflow.log_param("warmup_steps", 500)
                mlflow.log_param("weight_decay", 0.01)
                mlflow.log_param("train_pairs", len(self.train_dataset))
                mlflow.log_param("val_pairs", len(self.val_dataset))
                mlflow.log_param("test_pairs", len(self.test_dataset))
                mlflow.log_param("optimizer", "AdamW")
                mlflow.log_param("fp16", torch.cuda.is_available())
                
                # Log dataset info
                mlflow.log_param("dataset_name", "Food-101-Recipe-Pairs")
                mlflow.log_param("dataset_version", "1.0")
                
                # Tags
                mlflow.set_tag("stage", "training")
                mlflow.set_tag("model_type", "vision-language")
                mlflow.set_tag("task", "image-text-matching")
                mlflow.set_tag("status", "in_progress")
                
                # Custom callback for logging metrics
                from transformers import TrainerCallback
                
                class MLflowLoggingCallback(TrainerCallback):
                    def __init__(self):
                        self.step = 0
                        
                    def on_log(self, args, state, control, logs=None, **kwargs):
                        """Log metrics at each logging step"""
                        if logs:
                            self.step = state.global_step
                            
                            # Log training metrics
                            if 'loss' in logs:
                                mlflow.log_metric("train_loss", logs['loss'], step=self.step)
                            if 'learning_rate' in logs:
                                mlflow.log_metric("learning_rate", logs['learning_rate'], step=self.step)
                            
                            # Log eval metrics
                            if 'eval_loss' in logs:
                                mlflow.log_metric("eval_loss", logs['eval_loss'], step=self.step)
                    
                    def on_epoch_end(self, args, state, control, **kwargs):
                        """Log at end of each epoch"""
                        mlflow.log_metric("epoch", state.epoch, step=self.step)
                    
                    def on_train_end(self, args, state, control, **kwargs):
                        """Mark training as complete"""
                        mlflow.set_tag("status", "completed")
                
                # Initialize trainer with callback
                trainer = SigLIPTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.val_dataset,
                    callbacks=[MLflowLoggingCallback()]
                )
                
                # Train
                print("\n" + "="*60)
                print("Starting fine-tuning...")
                print(f"MLflow Run ID: {run_id}")
                print("="*60 + "\n")
                
                start_time = time.time()
                
                train_result = trainer.train()
                
                training_time = time.time() - start_time
                
                # Log final metrics
                mlflow.log_metric("final_train_loss", train_result.training_loss)
                mlflow.log_metric("training_time_seconds", training_time)
                mlflow.log_metric("training_time_minutes", training_time / 60)
                
                # Evaluate on validation
                print("\nEvaluating on validation set...")
                eval_result = trainer.evaluate()
                
                mlflow.log_metric("final_val_loss", eval_result['eval_loss'])
                
                # Evaluate on test
                print("\nEvaluating on test set...")
                test_loss = self._evaluate_test_set()
                mlflow.log_metric("final_test_loss", test_loss)
                
                # Log model artifacts
                print("\nSaving model artifacts...")
                temp_model_dir = self.output_dir / "temp_mlflow"
                temp_model_dir.mkdir(parents=True, exist_ok=True)
                
                self.model.save_pretrained(temp_model_dir)
                self.processor.save_pretrained(temp_model_dir)
                
                # Log model files
                mlflow.log_artifacts(str(temp_model_dir), "model")
                
                # Create and log training summary
                summary = {
                    "model": self.model_name,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "train_pairs": len(self.train_dataset),
                    "final_train_loss": float(train_result.training_loss),
                    "final_val_loss": float(eval_result['eval_loss']),
                    "final_test_loss": float(test_loss),
                    "training_time_minutes": training_time / 60,
                    "improvement": "TBD - compare with baseline"
                }
                
                summary_path = self.output_dir / "training_summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                mlflow.log_artifact(str(summary_path), "summary")
                
                # Generate training curves
                self._generate_training_curves(run_id)
                
                print(f"\n✓ Training complete!")
                print(f"  Training time: {training_time/60:.1f} minutes")
                print(f"  Final train loss: {train_result.training_loss:.4f}")
                print(f"  Final val loss: {eval_result['eval_loss']:.4f}")
                print(f"  Final test loss: {test_loss:.4f}")
                print(f"\n📊 View in MLflow: {run.info.artifact_uri}")
                
                # Cleanup temp directory
                shutil.rmtree(temp_model_dir)
        
        else:
            # Train without MLflow
            trainer = SigLIPTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
            )
            
            print("\n" + "="*60)
            print("Starting fine-tuning...")
            print("="*60 + "\n")
            
            trainer.train()
        
        # Save final model
        self.save_model()
        
        return trainer
    
    def save_model(self):
        """Save fine-tuned model"""
        print(f"\nSaving model to {self.output_dir}...")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        
        print(f"✓ Model saved to {self.output_dir}")
    
    def _evaluate_test_set(self):
        """Evaluate on test set and return loss"""
        self.model.eval()
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Prepare inputs - only pass keys that exist
                model_inputs = {
                    'pixel_values': batch['pixel_values'],
                    'input_ids': batch['input_ids'],
                    'return_loss': True
                }
                
                # Add attention_mask if present
                if 'attention_mask' in batch:
                    model_inputs['attention_mask'] = batch['attention_mask']
                
                outputs = self.model(**model_inputs)
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate_on_test(self):
        """Evaluate on test set"""
        print("\n" + "="*60)
        print("Evaluating on test set...")
        print("="*60 + "\n")
        
        test_loss = self._evaluate_test_set()
        
        print(f"\n✓ Test evaluation complete!")
        print(f"  Test loss: {test_loss:.4f}")
        
        return test_loss
    
    def _generate_training_curves(self, run_id):
        """Generate and log training curve visualizations"""
        import matplotlib.pyplot as plt
        
        # Get metrics from MLflow
        client = mlflow.tracking.MlflowClient()
        
        try:
            # Fetch metrics
            train_loss = client.get_metric_history(run_id, "train_loss")
            eval_loss = client.get_metric_history(run_id, "eval_loss")
            lr = client.get_metric_history(run_id, "learning_rate")
            
            if not train_loss:
                print("No training metrics to plot")
                return
            
            # Create figure with subplots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Training/Validation Loss
            train_steps = [m.step for m in train_loss]
            train_values = [m.value for m in train_loss]
            
            axes[0].plot(train_steps, train_values, label='Train Loss', linewidth=2)
            
            if eval_loss:
                eval_steps = [m.step for m in eval_loss]
                eval_values = [m.value for m in eval_loss]
                axes[0].plot(eval_steps, eval_values, label='Validation Loss', 
                            linewidth=2, linestyle='--')
            
            axes[0].set_xlabel('Training Steps', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Learning Rate Schedule
            if lr:
                lr_steps = [m.step for m in lr]
                lr_values = [m.value for m in lr]
                
                axes[1].plot(lr_steps, lr_values, color='orange', linewidth=2)
                axes[1].set_xlabel('Training Steps', fontsize=12)
                axes[1].set_ylabel('Learning Rate', fontsize=12)
                axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, 'No LR data', ha='center', va='center', 
                            fontsize=14, transform=axes[1].transAxes)
            
            plt.tight_layout()
            
            # Save and log
            curves_path = self.output_dir / "training_curves.png"
            plt.savefig(curves_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            mlflow.log_artifact(str(curves_path), "charts")
            print(f"✓ Training curves saved and logged")
            
        except Exception as e:
            print(f"Warning: Could not generate training curves: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune SigLIP on Food-101-Recipe-Pairs")
    parser.add_argument("--model", type=str, default="google/siglip-base-patch16-224",
                       help="Base model to fine-tune")
    parser.add_argument("--output-dir", type=str, default="./models/siglip-food-finetuned",
                       help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-6,
                       help="Learning rate")
    parser.add_argument("--no-mlflow", action="store_true",
                       help="Disable MLflow logging")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only evaluate existing model")
    
    args = parser.parse_args()
    
    # Initialize fine-tuner
    finetuner = SigLIPFineTuner(
        model_name=args.model,
        output_dir=args.output_dir
    )
    
    # Load model
    finetuner.load_model()
    
    # Prepare datasets
    finetuner.prepare_datasets()
    
    if args.eval_only:
        # Just evaluate
        finetuner.evaluate_on_test()
    else:
        # Fine-tune
        finetuner.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_mlflow=not args.no_mlflow
        )
    
    print("\n✓ All done!")


if __name__ == "__main__":
    main()