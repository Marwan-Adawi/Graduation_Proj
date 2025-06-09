import torch
import gc
import re
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from typing import List, Union, Optional
import numpy as np


class ArabicOCR:
    """
    A class for performing OCR on Arabic text images using the Qari-OCR model.
    
    This class provides memory-optimized processing for Arabic text extraction,
    with support for poetry, columned text, and proper right-to-left reading.
    """
    
    def __init__(self, 
                 model_name: str = "./Qari-OCR-0.3-SNAPSHOT-VL-2B-Instruct-merged",
                 cache_dir: str = "./cache",
                 device: Optional[str] = None,
                 max_size: int = 1024):
        """
        Initialize the Arabic OCR model.
        
        Args:
            model_name: HuggingFace model identifier
            cache_dir: Directory to cache the model and processor
            device: Device to run the model on ('cuda', 'mps', 'cpu'). Auto-detected if None.
            max_size: Default maximum image size for processing
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device or self._get_device()
        self.default_max_size = max_size
        
        # Initialize model and processor
        self.processor = None
        self.model = None
        self._load_model()
    
    def _get_device(self) -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Load the model and processor."""
        print(f"Loading model on {self.device}...")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, 
            cache_dir=f"{self.cache_dir}/processor"
        )
        
        # Load model
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name, 
            cache_dir=f"{self.cache_dir}/model"
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        print(f"Model loaded successfully on {self.device}")
    
    def _get_ocr_prompt(self) -> str:
        """Get the OCR instruction prompt for Arabic text."""
        return """
        You are an Arabic OCR specialist designed to extract text from images containing Arabic content. When processing images:

        1. ALWAYS read Arabic text from right to left, and top to bottom.
        2. When encountering poetry or columned text, process the rightmost column/section first before moving to columns/sections to the left.
        3. Preserve the exact format of poetry, maintaining the traditional two-hemistich structure with appropriate spacing.
        4. DO NOT hallucinate text when encountering:
        - Decorative elements (dots, lines, ornamental borders)
        - Page numbers, margin notes, or stamps
        - Blurry or partially visible text
        5. When text is unclear, mark it as [غير واضح] rather than guessing.
        6. Transcribe only what you can clearly see - accuracy is more important than completeness.
        7. Differentiate between actual numerical content and decorative elements that resemble numbers.
        8. NEVER generate emojis or non-Arabic symbols that aren't in the original text.
        9. For poetry, maintain the meter and rhyme structure as visible in the image.
        10. Ignore watermarks or background elements not part of the main text.
        """
    
    def _resize_image(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        """
        Resize image if it's too large to prevent memory issues.
        
        Args:
            image: PIL Image to resize
            max_size: Maximum dimension size
            
        Returns:
            Resized PIL Image
        """
        width, height = image.size
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        return image
    
    def _clear_memory(self):
        """Clear GPU/CPU memory and run garbage collection."""
        gc.collect()
        if hasattr(self, 'device') and self.device and self.device.startswith('cuda'):
            torch.cuda.empty_cache()
    
    def _process_single_image(self, 
                            image_input: Union[np.ndarray, Image.Image], 
                            max_size: int = 1024,
                            max_new_tokens= 3000):
        """
        Process a single image for OCR.
        
        Args:
            image_input: Numpy array or PIL Image
            max_size: Maximum image dimension
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Extracted text string
        """
        try:
            # Clear cache before processing
            self._clear_memory()
            
            # Convert to PIL Image if numpy array, otherwise use as is
            if isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input.copy()  # Create a copy to avoid modifying original
            else:
                raise ValueError(f"Unsupported image type: {type(image_input)}")
            
            # Resize if necessary
            image = self._resize_image(image, max_size)
            
            # Create messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self._get_ocr_prompt()}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Extract images from messages
            images = []
            for message in messages:
                if isinstance(message.get("content"), list):
                    for content in message["content"]:
                        if content.get("type") == "image":
                            images.append(content["image"])
            
            # Process inputs
            inputs = self.processor(
                text=[text], 
                images=images if images else None,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    temperature=0.001,        # Even more deterministic
                    do_sample=False,         # Keep greedy for accuracy
                    max_new_tokens=1500,     # Reduced from 2000
                    repetition_penalty=1.2,  # Increased from 1.3
                    length_penalty=0.9,      # Favor shorter outputs

                    early_stopping=True,     # Stop at natural end
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            result = output_text[0]
            
            # Clean up variables
            del inputs, generated_ids, generated_ids_trimmed, output_text, image, images, messages
            self._clear_memory()
            
            return result
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM error, trying with smaller image size...")
                self._clear_memory()
                
                # Retry with smaller image
                try:
                    return self._process_single_image(
                        image_input, 
                        max_size=512, 
                        max_new_tokens=2500
                    )
                except Exception:
                    return "[Error: Could not process image due to memory constraints]"
            else:
                raise e
    
    def extract_text(self, 
                    images: Union[List[Union[np.ndarray, Image.Image]], np.ndarray, Image.Image],
                    max_size: Optional[int] = None,
                    max_new_tokens: int = 1500,
                    clean_html: bool = True) -> Union[List[str], str]:
        """
        Extract text from one or multiple images.
        
        Args:
            images: Single image (numpy array or PIL Image) or list of images
            max_size: Maximum image dimension for processing (uses default if None)
            max_new_tokens: Maximum tokens to generate per image
            clean_html: Whether to remove HTML tags from output
            
        Returns:
            Extracted text(s) as string or list of strings
        """
        # Use default max_size if not specified
        if max_size is None:
            max_size = self.default_max_size
            
        # Handle single image (numpy array or PIL Image)
        if isinstance(images, (np.ndarray, Image.Image)):
            text = self._process_single_image(images, max_size, max_new_tokens)
            return self.clean_text(text) if clean_html else text
        
        # Handle multiple images
        if not isinstance(images, list):
            images = [images]
        
        extracted_texts = []
        total_images = len(images)
        
        for i, image_input in enumerate(images):
            print(f"Processing image {i+1}/{total_images}...")
            
            text = self._process_single_image(image_input, max_size, max_new_tokens)
            
            if clean_html:
                text = self.clean_text(text)
            
            extracted_texts.append(text)
            print(f"Extracted {len(text)} characters from image {i+1}")
        
        return extracted_texts
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean OCR text by removing HTML tags and extra spaces.
        
        Args:
            text: Raw OCR text output
            
        Returns:
            Cleaned text
        """
        # Remove all HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def process_batch(self, 
                     images: List[Union[np.ndarray, Image.Image]],
                     batch_size: int = 1,
                     **kwargs) -> List[str]:
        """
        Process images in batches (currently processes one at a time for memory efficiency).
        
        Args:
            images: List of image arrays or PIL Images
            batch_size: Batch size (currently unused, kept for future compatibility)
            **kwargs: Additional arguments passed to extract_text
            
        Returns:
            List of extracted texts
        """
        # For now, process one at a time for memory efficiency
        return self.extract_text(images, **kwargs)
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'processor') and self.processor is not None:
                del self.processor
            self._clear_memory()
        except (AttributeError, Exception):
            # Ignore errors during cleanup
            pass


