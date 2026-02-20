from PyQt6.QtCore import QThread, pyqtSignal
from PIL import Image
import numpy as np
import os
import csv

class WDTaggerWorker(QThread):
    finished = pyqtSignal(list, str) # tags, error_msg
    progress = pyqtSignal(str)
    
    def __init__(self, image_path, threshold=0.35):
        super().__init__()
        self.image_path = image_path
        self.threshold = threshold
        self.model_repo = "SmilingWolf/wd-vit-tagger-v3"
        self.model_name = "model.onnx"
        self.csv_name = "selected_tags.csv"
        
    def run(self):
        try:
            from huggingface_hub import hf_hub_download
            import onnxruntime as rt
        except ImportError:
            self.finished.emit([], "Required libraries (huggingface_hub, onnxruntime) are not installed.")
            return
            
        try:
            self.progress.emit("Downloading/Loading WD Tagger model...")
            model_path = hf_hub_download(self.model_repo, self.model_name)
            csv_path = hf_hub_download(self.model_repo, self.csv_name)
            
            self.progress.emit("Loading tags...")
            tags = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader) # skip header
                for row in reader:
                    tags.append(row[1]) # tag name is usually in the 2nd column
            
            self.progress.emit("Processing image...")
            image = Image.open(self.image_path).convert('RGB')
            # Preprocess image for WD Tagger (usually 448x448, BGR, etc. depending on v3)
            # WD14 vit-v3 uses 448x448
            image = image.resize((448, 448), Image.Resampling.BICUBIC)
            image_np = np.array(image, dtype=np.float32)
            # Normalize usually BGR or RGB depending on the specific model
            # For WD-v3, it is RGB, uint8 to float32, BGR->RGB 
            image_np = image_np[:, :, ::-1] # RGB to BGR (commonly used in cv2/onnx for wd)
            image_np = np.expand_dims(image_np, axis=0) # Add batch dim

            self.progress.emit("Running inference...")
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in rt.get_available_providers():
                providers = ['CUDAExecutionProvider'] + providers
                
            session = rt.InferenceSession(model_path, providers=providers)
            input_name = session.get_inputs()[0].name
            
            # The input might need to be (1, 448, 448, 3) float32
            probs = session.run(None, {input_name: image_np})[0]
            
            # Extract tags that exceed threshold
            result_tags = []
            for i, p in enumerate(probs[0]):
                if i < len(tags) and p >= self.threshold:
                    result_tags.append(tags[i])
                    
            self.finished.emit(result_tags, "")
        except Exception as e:
            self.finished.emit([], str(e))

class Florence2Worker(QThread):
    finished = pyqtSignal(list, str) # tags, error_msg
    progress = pyqtSignal(str)

    def __init__(self, image_path, task_prompt="<DETAILED_CAPTION>"):
        super().__init__()
        self.image_path = image_path
        self.task_prompt = task_prompt
        self.model_id = "microsoft/Florence-2-large"

    def run(self):
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            from unittest.mock import patch
            from transformers.dynamic_module_utils import get_imports
            import torch
        except ImportError:
            self.finished.emit([], "Required libraries (transformers, torch) are not installed.")
            return

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.progress.emit(f"Loading Florence-2 model on {device} (may take a while)...")
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
                if not str(filename).endswith("modeling_florence2.py"):
                    return get_imports(filename)
                imports = get_imports(filename)
                if "flash_attn" in imports:
                    imports.remove("flash_attn")
                return imports

            with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, 
                    torch_dtype=torch_dtype, 
                    trust_remote_code=True
                ).to(device)
                processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

            self.progress.emit("Processing image...")
            image = Image.open(self.image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            inputs = processor(text=self.task_prompt, images=image, return_tensors="pt").to(device, torch_dtype)

            self.progress.emit("Generating caption...")
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            parsed_answer = processor.post_process_generation(generated_text, task=self.task_prompt, image_size=(image.width, image.height))
            
            result = parsed_answer.get(self.task_prompt, "")
            
            # Florence returns a sentence. We can treat it as one large tag or split it.
            # We'll return it as a single tag for the tag editor.
            result_tags = [result.strip()] if result else []
            self.finished.emit(result_tags, "")

        except Exception as e:
            self.finished.emit([], str(e))
