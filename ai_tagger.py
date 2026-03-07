import os
import csv
import traceback
import sys
from PyQt6.QtCore import QThread, pyqtSignal
from PIL import Image

def get_onnx_device():
    try:
        import onnxruntime as rt
        available_providers = rt.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            return "GPU (CUDA)", available_providers
        elif 'DmlExecutionProvider' in available_providers:
            return "GPU (DirectML)", available_providers
        else:
            return "CPU", available_providers
    except ImportError:
        return None, []

def get_torch_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

class PixAITaggerWorker(QThread):
    finished = pyqtSignal(list, str) # tags, error_msg
    progress = pyqtSignal(str)

    def __init__(self, image_path, threshold=0.35):
        super().__init__()
        self.image_path = image_path
        self.threshold = threshold
        self.model_name = "v0.9"

    def run(self):
        print(f"--- Starting PixAI Tagger ---")
        device_name, providers = get_onnx_device()
        
        if device_name is None:
            self.finished.emit([], "ONNX Runtime not installed.")
            return

        try:
            self.progress.emit(f"Running inference on {device_name}...")
            from imgutils.tagging.pixai import get_pixai_tags
            import inspect
            
            sig = inspect.signature(get_pixai_tags)
            params = sig.parameters
            
            tagger_kwargs = {"model_name": self.model_name}
            if "threshold" in params:
                tagger_kwargs["threshold"] = self.threshold
            elif "thresholds" in params:
                tagger_kwargs["thresholds"] = self.threshold

            general_tags, character_tags = get_pixai_tags(self.image_path, **tagger_kwargs)
            result_tags = list(character_tags.keys()) + list(general_tags.keys())
            self.finished.emit(result_tags, "")
        except (ImportError, TypeError):
            print("PixAI module not found or incompatible. Falling back to SwinV2...")
            try:
                from imgutils.tagging import get_wd14_tags
                general_tags, character_tags = get_wd14_tags(
                    self.image_path, model_name='SwinV2',
                    general_threshold=self.threshold, character_threshold=self.threshold
                )
                result_tags = list(character_tags.keys()) + list(general_tags.keys())
                self.finished.emit(result_tags, "")
            except Exception as e:
                self.finished.emit([], str(e))
        except Exception as e:
            traceback.print_exc()
            self.finished.emit([], str(e))

class BatchPixAITaggerWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(int, int, str)

    def __init__(self, file_manager, image_paths, threshold=0.35):
        super().__init__()
        self.file_manager = file_manager
        self.image_paths = image_paths
        self.threshold = threshold
        self.model_name = "v0.9"

    def run(self):
        device_name, _ = get_onnx_device()
        if device_name is None:
            self.finished.emit(0, len(self.image_paths), "ONNX Runtime not installed.")
            return

        success_count = 0
        total = len(self.image_paths)
        use_fallback = False
        
        try:
            from imgutils.tagging.pixai import get_pixai_tags
            import inspect
            sig = inspect.signature(get_pixai_tags)
            params = sig.parameters
        except ImportError:
            use_fallback = True
            from imgutils.tagging import get_wd14_tags

        for i, img_path in enumerate(self.image_paths):
            if self.isInterruptionRequested(): break
            self.progress.emit(i, total, os.path.basename(img_path))
            try:
                if not use_fallback:
                    tagger_kwargs = {"model_name": self.model_name}
                    if "threshold" in params:
                        tagger_kwargs["threshold"] = self.threshold
                    elif "thresholds" in params:
                        tagger_kwargs["thresholds"] = self.threshold
                        
                    general_tags, character_tags = get_pixai_tags(img_path, **tagger_kwargs)
                else:
                    general_tags, character_tags = get_wd14_tags(img_path, model_name='SwinV2', general_threshold=self.threshold, character_threshold=self.threshold)
                
                new_tags = list(character_tags.keys()) + list(general_tags.keys())
                current_tags = self.file_manager.read_tags(img_path)
                added = False
                for tag in new_tags:
                    if tag not in current_tags:
                        current_tags.append(tag)
                        added = True
                if added: self.file_manager.save_tags(img_path, current_tags)
                success_count += 1
            except Exception as e:
                print(f"Error: {e}")
        self.finished.emit(success_count, total, "")

class Florence2Worker(QThread):
    finished = pyqtSignal(list, str)
    progress = pyqtSignal(str)

    def __init__(self, image_path, task_prompt="<DETAILED_CAPTION>"):
        super().__init__()
        self.image_path = image_path
        self.task_prompt = task_prompt
        self.model_id = "microsoft/Florence-2-base"

    def run(self):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
            from unittest.mock import patch
            from transformers.dynamic_module_utils import get_imports

            device = get_torch_device()
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            if device == "cuda": torch.cuda.empty_cache()

            def fixed_get_imports(filename):
                imports = get_imports(filename)
                if "flash_attn" in imports: imports.remove("flash_attn")
                return imports

            config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
            if not hasattr(config, "forced_bos_token_id"): config.forced_bos_token_id = None
            
            with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
                model = AutoModelForCausalLM.from_pretrained(self.model_id, config=config, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
                processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

            image = Image.open(self.image_path).convert("RGB")
            inputs = processor(text=self.task_prompt, images=image, return_tensors="pt").to(device, torch_dtype)
            
            generated_ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(generated_text, task=self.task_prompt, image_size=image.size)
            
            result = parsed_answer.get(self.task_prompt, "")
            self.finished.emit([result.strip()] if result else [], "")
        except Exception as e:
            traceback.print_exc()
            self.finished.emit([], str(e))

class BatchFlorence2Worker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(int, int, str)

    def __init__(self, file_manager, image_paths, task_prompt="<DETAILED_CAPTION>"):
        super().__init__()
        self.file_manager = file_manager
        self.image_paths = image_paths
        self.task_prompt = task_prompt
        self.model_id = "microsoft/Florence-2-base"

    def run(self):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
            from unittest.mock import patch
            from transformers.dynamic_module_utils import get_imports

            device = get_torch_device()
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            if device == "cuda": torch.cuda.empty_cache()

            def fixed_get_imports(filename):
                imports = get_imports(filename)
                if "flash_attn" in imports: imports.remove("flash_attn")
                return imports

            config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
            if not hasattr(config, "forced_bos_token_id"): config.forced_bos_token_id = None

            with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
                model = AutoModelForCausalLM.from_pretrained(self.model_id, config=config, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
                processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

            success_count = 0
            for i, img_path in enumerate(self.image_paths):
                if self.isInterruptionRequested(): break
                self.progress.emit(i, len(self.image_paths), os.path.basename(img_path))
                try:
                    image = Image.open(img_path).convert("RGB")
                    inputs = processor(text=self.task_prompt, images=image, return_tensors="pt").to(device, torch_dtype)
                    generated_ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                    parsed_answer = processor.post_process_generation(generated_text, task=self.task_prompt, image_size=image.size)
                    result = parsed_answer.get(self.task_prompt, "").strip()
                    
                    if result:
                        tags = self.file_manager.read_tags(img_path)
                        if result not in tags:
                            tags.append(result)
                            self.file_manager.save_tags(img_path, tags)
                    success_count += 1
                except Exception:
                    traceback.print_exc()
            self.finished.emit(success_count, len(self.image_paths), "")
        except Exception as e:
            self.finished.emit(0, len(self.image_paths), str(e))
