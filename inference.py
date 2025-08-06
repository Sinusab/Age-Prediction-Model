import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
import json

from config import config
from models import create_model
from dataset import TransformFactory
from utils import load_checkpoint, print_system_info

class AgePredictor:
    """Class for age prediction from images"""
    
    def __init__(self, model_path, config_obj=None):
        """
        Args:
            model_path: Path to trained model file
            config_obj: Configuration (optional)
        """
        self.config = config_obj or config
        self.device = self.config.device
        
        # Load model
        self.model = None
        self.mean_per_channel = None
        self.std_per_channel = None
        self.transform = None
        
        self.load_model(model_path)
        
        print(f"✅ Model loaded. Device: {self.device}")
    
    def load_model(self, model_path):
        """Load model from checkpoint"""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"📂 Loading model from {model_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        self.model = create_model('single_task', 'improved_cnn', pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load normalization statistics
        self.mean_per_channel = checkpoint.get('mean_per_channel')
        self.std_per_channel = checkpoint.get('std_per_channel')
        
        if self.mean_per_channel is None or self.std_per_channel is None:
            print("⚠️ Normalization statistics not found in checkpoint. Using default values.")
            self.mean_per_channel = np.array([0.485, 0.456, 0.406])
            self.std_per_channel = np.array([0.229, 0.224, 0.225])
        
        # Create transform
        self.transform = TransformFactory.get_test_transforms(
            self.mean_per_channel, self.std_per_channel
        )
        
        # Print model information
        epoch = checkpoint.get('epoch', 'Unknown')
        val_loss = checkpoint.get('val_loss', 'Unknown')
        val_mae = checkpoint.get('val_mae', 'Unknown')
        
        print(f"📊 Model Information:")
        print(f"  Epoch: {epoch}")
        print(f"  Validation Loss: {val_loss}")
        print(f"  Validation MAE: {val_mae}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transform
            image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            
            return image_tensor
            
        except Exception as e:
            raise ValueError(f"Error processing image {image_path}: {str(e)}")
    
    def predict_single_image(self, image_path, sex, race):
        """
        Predict age for a single image
        
        Args:
            image_path: Path to image
            sex: Gender (0 = male, 1 = female)
            race: Race (0-4)
            
        Returns:
            dict: Prediction result
        """
        
        # Validate inputs
        if sex not in [0, 1]:
            raise ValueError("Gender must be 0 (male) or 1 (female)")
        
        if race not in [0, 1, 2, 3, 4]:
            raise ValueError("Race must be between 0 and 4")
        
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Create gender and race tensors
        sex_tensor = torch.zeros((1, 2), dtype=torch.float32, device=self.device)
        sex_tensor[0, sex] = 1.0
        
        race_tensor = torch.zeros((1, 5), dtype=torch.float32, device=self.device)
        race_tensor[0, race] = 1.0
        
        # Prediction
        with torch.no_grad():
            predicted_age = self.model(image_tensor, sex_tensor, race_tensor)
            predicted_age = predicted_age.item()
        
        return {
            'image_path': image_path,
            'predicted_age': round(predicted_age, 2),
            'sex': sex,
            'race': race,
            'sex_label': 'Female' if sex == 1 else 'Male',
            'race_label': f'Race {race}'
        }
    
    def predict_batch(self, image_paths, sexes, races):
        """
        Predict age for multiple images
        
        Args:
            image_paths: List of image paths
            sexes: List of genders
            races: List of races
            
        Returns:
            list: List of prediction results
        """
        
        if len(image_paths) != len(sexes) or len(image_paths) != len(races):
            raise ValueError("Number of images, genders, and races must be equal")
        
        results = []
        
        for img_path, sex, race in zip(image_paths, sexes, races):
            try:
                result = self.predict_single_image(img_path, sex, race)
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing {img_path}: {str(e)}")
                results.append({
                    'image_path': img_path,
                    'error': str(e)
                })
        
        return results
    
    def predict_interactive(self):
        """Interactive mode for prediction"""
        
        print("\n" + "="*50)
        print("🎯 Interactive Age Prediction Mode")
        print("="*50)
        print("Type 'exit' or 'quit' to exit")
        
        while True:
            try:
                # Get image path
                image_path = input("\n📸 Image path: ").strip()
                
                if image_path.lower() in ['exit', 'quit']:
                    print("👋 Exiting...")
                    break
                
                if not os.path.exists(image_path):
                    print("❌ File not found!")
                    continue
                
                # Get gender
                sex_input = input("👤 Gender (0=male, 1=female): ").strip()
                try:
                    sex = int(sex_input)
                    if sex not in [0, 1]:
                        raise ValueError()
                except:
                    print("❌ Invalid gender! Must be 0 or 1.")
                    continue
                
                # Get race
                race_input = input("🌍 Race (0-4): ").strip()
                try:
                    race = int(race_input)
                    if race not in [0, 1, 2, 3, 4]:
                        raise ValueError()
                except:
                    print("❌ Invalid race! Must be between 0 and 4.")
                    continue
                
                # Prediction
                print("🔄 Predicting...")
                result = self.predict_single_image(image_path, sex, race)
                
                # Display result
                print("\n" + "="*30)
                print("📊 Prediction Result:")
                print("="*30)
                print(f"📸 Image: {result['image_path']}")
                print(f"🎂 Predicted Age: {result['predicted_age']} years")
                print(f"👤 Gender: {result['sex_label']}")
                print(f"🌍 Race: {result['race_label']}")
                print("="*30)
                
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Age prediction from images')
    
    parser.add_argument('--model', '-m', type=str, required=True,
                       help='Path to model file (.pth)')
    
    parser.add_argument('--image', '-i', type=str,
                       help='Path to image for prediction')
    
    parser.add_argument('--sex', '-s', type=int, choices=[0, 1],
                       help='Gender (0=male, 1=female)')
    
    parser.add_argument('--race', '-r', type=int, choices=[0, 1, 2, 3, 4],
                       help='Race (0-4)')
    
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')
    
    parser.add_argument('--batch', type=str,
                       help='JSON file containing list of images for batch processing')
    
    args = parser.parse_args()
    
    # Print system information
    print_system_info()
    
    try:
        # Create predictor
        predictor = AgePredictor(args.model)
        
        if args.interactive:
            # Interactive mode
            predictor.predict_interactive()
            
        elif args.batch:
            # Batch processing
            print(f"📂 Loading batch file: {args.batch}")
            
            with open(args.batch, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
            
            results = predictor.predict_batch(
                batch_data['images'],
                batch_data['sexes'],
                batch_data['races']
            )
            
            # Save results
            output_file = args.batch.replace('.json', '_results.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Results saved to {output_file}")
            
        elif args.image and args.sex is not None and args.race is not None:
            # Single image prediction
            result = predictor.predict_single_image(args.image, args.sex, args.race)
            
            print("\n" + "="*50)
            print("📊 Prediction Result:")
            print("="*50)
            print(f"📸 Image: {result['image_path']}")
            print(f"🎂 Predicted Age: {result['predicted_age']} years")
            print(f"👤 Gender: {result['sex_label']}")
            print(f"🌍 Race: {result['race_label']}")
            print("="*50)
            
        else:
            print("❌ Please select one of the following options:")
            print("  - Interactive mode: --interactive")
            print("  - Single image: --image <path> --sex <0|1> --race <0-4>")
            print("  - Batch processing: --batch <json_file>")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()