#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Native Agent service for LLM integration using PyTorch models directly."""

import logging
import random
import gc
import time
from enum import Enum
from typing import Optional, List, Dict, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config

# Set up logging
logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
NATIVE_LLM_MODEL = config.NATIVE_LLM_MODEL
NATIVE_LLM_PRECISION = config.NATIVE_LLM_PRECISION
NATIVE_LLM_MAX_NEW_TOKENS = config.NATIVE_LLM_MAX_NEW_TOKENS
NATIVE_LLM_DEVICE = config.NATIVE_LLM_DEVICE


def is_quantized_precision(precision: str) -> bool:
    """Check if precision string indicates INT4 GPTQ quantization."""
    return precision.lower() in ["int4", "gptq", "int4_gptq"]


def get_torch_dtype(precision: str) -> torch.dtype:
    """Convert precision string to torch dtype."""
    # For GPTQ INT4 models, use bfloat16 as the compute dtype
    # (weights are INT4, but activations use bfloat16 for stability)
    if is_quantized_precision(precision):
        return torch.bfloat16
    
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return dtype_map.get(precision.lower(), torch.bfloat16)


def get_gpu_memory_info(device: str = "cuda:0") -> Dict[str, float]:
    """Get GPU memory information in GB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "total": 0, "free": 0}
    
    device_idx = int(device.split(":")[-1]) if ":" in device else 0
    
    allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 3)  # GB
    reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 3)  # GB
    total = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)  # GB
    free = total - allocated
    
    return {
        "allocated": round(allocated, 2),
        "reserved": round(reserved, 2),
        "total": round(total, 2),
        "free": round(free, 2)
    }


def log_gpu_memory(prefix: str = "", device: str = "cuda:0"):
    """Log current GPU memory usage (only if VERBOSE is enabled)."""
    if config.VERBOSE and torch.cuda.is_available():
        mem_info = get_gpu_memory_info(device)
        logger.info(f"{prefix}GPU Memory - Allocated: {mem_info['allocated']:.2f} GB, "
                   f"Reserved: {mem_info['reserved']:.2f} GB, "
                   f"Free: {mem_info['free']:.2f} GB, "
                   f"Total: {mem_info['total']:.2f} GB")


class RuleType(Enum):
    """Enum for different rule types."""
    PLANNING = "planning"
    PROMPT_GENERATION = "prompt_generation"
    INPUT_CLASSIFICATION = "input_classification"


class NativeLLMAgent:
    """Native LLM agent using PyTorch and HuggingFace transformers."""
    
    def __init__(self, load_model: bool = True):
        """Initialize the native LLM agent.
        
        Args:
            load_model: If True, load the model immediately. If False, defer loading
                       until ensure_model_loaded() is called. This allows for 
                       coordinated loading with other models via GPUMemoryManager.
        """
        self.model = None
        self.tokenizer = None
        self.device = NATIVE_LLM_DEVICE
        self.is_loaded = False
        self.conversation_history: List[Dict[str, str]] = []
        self.is_generating_prompts = False
        
        if load_model:
            self._load_model()
    
    def ensure_model_loaded(self):
        """Ensure the model is loaded. Call this before any inference."""
        if not self.is_loaded:
            self._load_model()
    
    def _get_input_classification_rules(self) -> str:
        """Get rules for input classification phase."""
        return """You are an input classifier for a 3D scene creation application.
Your task is to determine if the user's input is a scene description or something else.
A scene description should describe a physical environment, location, or setting that can contain objects.
Scene descriptions typically include: locations (beach, kitchen, garden), environments (outdoor, indoor), settings (modern, rustic, tropical), or specific places.
Non-scene inputs include: greetings (hello, hi), questions (what can you do, how does this work), general chat (thanks, nice app), or requests for help.

You must respond with exactly one of these classifications:
1. 'SCENE' - if the input describes a scene, environment, or location that could contain objects
2. 'GREETING' - if the input is a greeting or salutation
3. 'QUESTION' - if the input is asking about capabilities, how to use the app, or seeking information
4. 'GENERAL_CHAT' - if the input is general conversation, thanks, or other non-scene content

Respond with only the classification word (SCENE, GREETING, QUESTION, or GENERAL_CHAT) - no additional text or explanation."""
    
    def _get_planning_rules(self) -> str:
        """Get rules for scene planning phase."""
        return f"""You are a helpful scene planning assistant for 3D content creation.
Your primary task is to suggest objects for the 3D scene based on user requests.
Always suggest objects in singular form (e.g., 'Palm Tree' instead of 'Palm Trees').
Always format object names with proper capitalization and spaces (e.g., 'Coffee Table' not 'coffee_table').
When suggesting objects for a general scene request: Suggest exactly {config.NUM_OF_OBJECTS} objects that would be appropriate for this scene.
Always format your object suggestions in this exact format: Suggested objects: 1. object_name 2. object_name ... {config.NUM_OF_OBJECTS}. object_name
Focus only on suggesting appropriate objects for the scene based on the user's requests."""
    
    def _get_prompt_generation_rules(self) -> str:
        """Get rules for 2D prompt generation phase."""
        return f"""You are now in 2D prompt generation mode. Your task is to create detailed prompts for each object in the scene.
The descriptions should be highly detailed and visually rich, suitable for a text-to-image generation model.
The prompt must specify a plain or empty background (e.g., 'on a white background', 'isolated', 'no background').
Focus ONLY on the physical and visual characteristics of each object itself.
Keep each object's prompt to exactly {config.TWO_D_PROMPT_LENGTH} words or less for optimal generation quality.
Generate a separate prompt for each object in the scene.
Format each object's prompt with 'Object:' and 'Prompt:' labels.
Do not add any explanatory notes or comments after the prompt.
Do not use asterisks or any special formatting characters.
Output only the Object and Prompt labels with clean text - no additional formatting or notes.

The prompt text should describe the object's visual characteristics in detail.
Example:
Object: Beach Chair
Prompt: A comfortable beach chair with ergonomic design and colorful fabric, on a white background
Example:
Object: Beach Umbrella
Prompt: A vibrant beach umbrella with colorful stripes and sturdy metal frame, on a white background"""
    
    def _load_model(self):
        """Load the native PyTorch LLM model (supports full precision and GPTQ INT4 quantized models)."""
        try:
            # Determine if using INT4 GPTQ quantization
            use_quantized = is_quantized_precision(NATIVE_LLM_PRECISION)
            
            # Use the configured model (same config for both regular and quantized)
            model_name = NATIVE_LLM_MODEL
            
            if use_quantized:
                logger.info(f"Loading INT4 quantized model: {model_name}")
            else:
                logger.info(f"Loading native LLM model: {model_name}")
            
            if config.VERBOSE:
                logger.info(f"Using precision: {NATIVE_LLM_PRECISION}")
                logger.info(f"Quantization: {'INT4' if use_quantized else 'None (full precision)'}")
                logger.info(f"Target device: {self.device}")
                # Log GPU memory before loading
                log_gpu_memory("Before model load - ", self.device)
            
            vram_before = get_gpu_memory_info(self.device)["allocated"] if config.VERBOSE else 0
            
            torch_dtype = get_torch_dtype(NATIVE_LLM_PRECISION)
            
            # Start timing
            load_start_time = time.time()
            
            # Load tokenizer
            tokenizer_start = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            tokenizer_time = time.time() - tokenizer_start
            if config.VERBOSE:
                logger.info(f"Tokenizer loaded in {tokenizer_time:.2f} seconds")
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model - different approach for quantized vs regular models
            model_start = time.time()
            
            if use_quantized:
                # For GPTQ models, transformers auto-detects the quantization config
                # from the model's config.json (requires auto-gptq to be installed)
                logger.info("Loading INT4 GPTQ quantized model")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map=self.device,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,  # Important for quantized models
                )
            else:
                # Standard model loading for full precision
                # Use torch_dtype (the warning is misleading - torch_dtype is correct for transformers)
                logger.info(f"Loading model with torch_dtype={torch_dtype}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                ).to(self.device)
                logger.info(f"Model moved to {self.device}")
            
            model_load_time = time.time() - model_start
            if config.VERBOSE:
                logger.info(f"Model weights loaded in {model_load_time:.2f} seconds")
            
            self.model.eval()
            self.is_loaded = True
            
            # Calculate total load time
            total_load_time = time.time() - load_start_time
            
            # Log verbose summary
            if config.VERBOSE:
                # Log GPU memory after loading
                log_gpu_memory("After model load - ", self.device)
                vram_after = get_gpu_memory_info(self.device)["allocated"]
                vram_used = vram_after - vram_before
                
                # Log summary
                logger.info(f"=" * 60)
                logger.info(f"Native LLM Model Load Summary:")
                logger.info(f"  Model: {model_name}")
                logger.info(f"  Precision: {NATIVE_LLM_PRECISION}")
                logger.info(f"  Quantization: {'INT4' if use_quantized else 'None'}")
                logger.info(f"  Device: {self.device}")
                logger.info(f"  Total load time: {total_load_time:.2f} seconds")
                logger.info(f"  VRAM used by model: {vram_used:.2f} GB")
                logger.info(f"=" * 60)
            else:
                quant_info = " (INT4)" if use_quantized else ""
                logger.info(f"Successfully loaded native LLM model{quant_info} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load native LLM model: {e}")
            self.is_loaded = False
            raise
    
    def _build_prompt(self, user_message: str, system_prompt: str) -> str:
        """Build the full prompt with system instructions and conversation history."""
        # Build messages in chat format
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in self.conversation_history:
            messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Use the tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # Disable thinking mode for reasoning models
            )
        else:
            # Fallback for models without chat template
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            for msg in self.conversation_history:
                role = msg["role"]
                content = msg["content"]
                prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return prompt
    
    def _generate(self, prompt: str, temperature: float = None) -> str:
        """Generate response from the model."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model is not loaded")
        
        if temperature is None:
            temperature = config.LLM_TEMPERATURE
        
        # Verify model is on correct device
        if config.VERBOSE:
            model_device = next(self.model.parameters()).device
            logger.info(f"Starting inference - Model on device: {model_device}")
        
        # Start timing for tokenization (only if verbose)
        tokenize_start = time.time() if config.VERBOSE else 0
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.device)
        
        tokenize_time = time.time() - tokenize_start if config.VERBOSE else 0
        input_tokens = inputs["input_ids"].shape[1] if config.VERBOSE else 0
        
        # Set seed for reproducibility if enabled
        if config.LLM_RANDOM_SEED_ENABLED:
            seed = random.randint(1, 999999)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Start timing for generation (only if verbose)
        generate_start = time.time() if config.VERBOSE else 0
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=NATIVE_LLM_MAX_NEW_TOKENS,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        
        generate_time = time.time() - generate_start if config.VERBOSE else 0
        
        # Decode response (only the new tokens)
        input_length = inputs["input_ids"].shape[1]
        
        decode_start = time.time() if config.VERBOSE else 0
        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        ).strip()
        decode_time = time.time() - decode_start if config.VERBOSE else 0
        
        # Log inference metrics (only if verbose)
        if config.VERBOSE:
            output_tokens = outputs[0].shape[0] - input_length
            tokens_per_second = output_tokens / generate_time if generate_time > 0 else 0
            total_time = tokenize_time + generate_time + decode_time
            
            logger.info(f"Inference metrics - Input tokens: {input_tokens}, Output tokens: {output_tokens}")
            logger.info(f"Inference time - Tokenize: {tokenize_time:.3f}s, Generate: {generate_time:.3f}s, Decode: {decode_time:.3f}s, Total: {total_time:.3f}s")
            logger.info(f"Generation speed: {tokens_per_second:.2f} tokens/second")
        
        return response
    
    def run(self, prompt: str, rule_type: RuleType = RuleType.PLANNING):
        """Run a prompt through the LLM with specified rule type."""
        try:
            # Get appropriate system prompt based on rule type
            if rule_type == RuleType.PROMPT_GENERATION:
                system_prompt = self._get_prompt_generation_rules()
                self.is_generating_prompts = True
            elif rule_type == RuleType.INPUT_CLASSIFICATION:
                system_prompt = self._get_input_classification_rules()
                self.is_generating_prompts = False
            else:
                system_prompt = self._get_planning_rules()
                self.is_generating_prompts = False
            
            # Build full prompt
            full_prompt = self._build_prompt(prompt, system_prompt)
            
            # Generate response
            response = self._generate(full_prompt)
            
            # Add to conversation history (for context in multi-turn conversations)
            if rule_type == RuleType.PLANNING:
                self.conversation_history.append({"role": "user", "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": response})
            
            # Return a simple response object
            return NativeAgentResponse(response)
            
        except Exception as e:
            logger.error(f"Error running prompt: {e}")
            raise
    
    def check_agent_health(self) -> bool:
        """Check if the LLM agent is loaded and ready."""
        return self.is_loaded and self.model is not None
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.conversation_history = []
        logger.info("Conversation memory cleared")
    
    def move_to_device(self, device: str):
        """Move model to specified device."""
        if self.model is not None:
            logger.info(f"Moving native LLM model to {device}")
            move_start = time.time() if config.VERBOSE else 0
            
            # Log memory before move (only if verbose)
            if config.VERBOSE and "cuda" in self.device:
                log_gpu_memory("Before move - ", self.device)
            
            self.model.to(device)
            self.device = device
            
            # Log memory after move and timing (only if verbose)
            if config.VERBOSE:
                move_time = time.time() - move_start
                if "cuda" in device:
                    log_gpu_memory("After move - ", device)
                logger.info(f"Successfully moved model to {device} in {move_time:.2f} seconds")
            else:
                logger.info(f"Successfully moved model to {device}")
    
    def move_to_cpu(self):
        """Move model to CPU to free GPU memory."""
        self.move_to_device("cpu")
        self._clear_gpu_memory()
    
    def move_to_gpu(self):
        """Move model back to GPU."""
        self.move_to_device(NATIVE_LLM_DEVICE)
    
    def _clear_gpu_memory(self):
        """Clear GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_vram_usage(self) -> Dict[str, float]:
        """Get current VRAM usage information."""
        return get_gpu_memory_info(self.device)
    
    def log_current_vram(self, prefix: str = "", force: bool = False):
        """Log current VRAM usage (respects VERBOSE setting unless force=True)."""
        if force or config.VERBOSE:
            # Temporarily enable logging even if VERBOSE is False
            if torch.cuda.is_available():
                mem_info = get_gpu_memory_info(self.device)
                logger.info(f"{prefix}GPU Memory - Allocated: {mem_info['allocated']:.2f} GB, "
                           f"Reserved: {mem_info['reserved']:.2f} GB, "
                           f"Free: {mem_info['free']:.2f} GB, "
                           f"Total: {mem_info['total']:.2f} GB")
    
    def cleanup(self):
        """Clean up the model and free memory."""
        if self.model is not None:
            logger.info("Cleaning up native LLM model...")
            try:
                self.model.cpu()
                del self.model
                self.model = None
            except Exception as e:
                logger.error(f"Error during model cleanup: {e}")
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.is_loaded = False
        self.device = "cpu"
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("LLM model cleanup complete")
        self._clear_gpu_memory()
        logger.info("Native LLM model cleanup complete")


class NativeAgentResponse:
    """Simple response wrapper to match the interface of the original agent."""
    
    def __init__(self, value: str):
        self.output = NativeAgentOutput(value)


class NativeAgentOutput:
    """Output wrapper for response value."""
    
    def __init__(self, value: str):
        self.value = value


class NativeAgentService:
    """Service class for managing native LLM agent interactions."""
    
    def __init__(self):
        """Initialize the native agent service with lazy loading.
        
        Note: Model is NOT loaded here. It will be loaded by GPUMemoryManager.preload_all_models()
        in the correct order (after TRELLIS and SANA are loaded and moved to CPU).
        """
        self.agent: Optional[NativeLLMAgent] = None
        # Don't load model here - let preload_all_models() handle the order
    
    def _initialize_agent(self, load_model: bool = True):
        """Initialize the planning agent.
        
        Args:
            load_model: If True, load the LLM model immediately. If False, defer
                       loading until explicitly called. Used by GPUMemoryManager 
                       to control loading order.
        """
        try:
            self.agent = NativeLLMAgent(load_model=load_model)
            if load_model:
                print("Native LLM agent initialized and model loaded", flush=True)
            else:
                print("Native LLM agent initialized (model loading deferred)", flush=True)
        except Exception as e:
            print(f"Failed to initialize native LLM agent: {e}", flush=True)
            raise
    
    def _ensure_agent_loaded(self, load_model: bool = True):
        """Ensure agent and model are loaded.
        
        Args:
            load_model: If True, also ensure the model is loaded. If False, only
                       create the agent wrapper without loading the model.
        """
        if self.agent is None:
            self._initialize_agent(load_model=load_model)
        elif load_model and not self.agent.is_loaded:
            # Agent exists but model not loaded - load it now
            self.agent.ensure_model_loaded()
    
    def is_healthy(self) -> bool:
        """Check if the agent is healthy and ready."""
        return self.agent is not None and self.agent.check_agent_health()
    
    def chat(self, message: str, current_objects=None) -> str:
        """Send a chat message to the agent."""
        try:
            self._ensure_agent_loaded()
            response = self.agent.run(message)
            return response.output.value
        except Exception as e:
            return f"Error communicating with agent: {str(e)}"
    
    def classify_input(self, user_input: str) -> Tuple[str, Optional[str]]:
        """Classify if user input is a scene description or something else."""
        try:
            if not user_input.strip():
                return "EMPTY", "Please enter a scene description."
            
            # Load agent on first use
            self._ensure_agent_loaded()
            
            # Use the LLM to classify the input
            response = self.agent.run(user_input, RuleType.INPUT_CLASSIFICATION)
            classification = response.output.value.strip().upper()
            
            # Validate the classification
            valid_classifications = ["SCENE", "GREETING", "QUESTION", "GENERAL_CHAT"]
            if classification not in valid_classifications:
                # Fallback to scene if classification is unclear
                classification = "SCENE"
            
            # Generate appropriate response messages
            if classification == "SCENE":
                return "SCENE", None  # No message needed, proceed with scene processing
            elif classification == "GREETING":
                message = "Hello! To get started, please describe a scene you'd like to create. For example: 'A cozy living room with a fireplace' or 'A tropical beach with palm trees'"
                return "GREETING", message
            elif classification == "QUESTION":
                message = "I can help you create 3D scenes! Describe what you want to build, like 'A modern kitchen' or 'A garden with flowers'. I'll generate objects for your scene and create 3D models."
                return "QUESTION", message
            elif classification == "GENERAL_CHAT":
                message = "Thanks! To create a 3D scene, please describe what you'd like to build. Try something like 'A beach scene' or 'A modern office'."
                return "GENERAL_CHAT", message
            else:
                return "SCENE", None  # Default fallback
                
        except Exception as e:
            logger.error(f"Error classifying input: {e}")
            return "SCENE", None  # Fallback to scene processing on error
    
    def generate_objects_for_scene(self, scene_description: str) -> List[str]:
        """Generate objects for a given scene description."""
        try:
            self._ensure_agent_loaded()
            
            # Create a specific prompt for generating objects
            prompt = f"""Based on this scene description: "{scene_description}"

Please suggest exactly {config.NUM_OF_OBJECTS} objects that would be appropriate for this scene. 

Requirements:
- Suggest exactly {config.NUM_OF_OBJECTS} objects (no more, no less)
- Use singular form for all objects (e.g., 'Chair' not 'Chairs')
- Use proper capitalization and spaces (e.g., 'Coffee Table' not 'coffee_table')
- Focus on objects that would realistically be found in this type of scene
- Ensure variety and complementarity between objects

Format your response as:
Suggested objects:
1. object_name
2. object_name
3. object_name
...
{config.NUM_OF_OBJECTS}. object_name

Scene arrangement: [Brief description of how these objects could be arranged together]"""

            response = self.agent.run(prompt, RuleType.PLANNING)
            response_text = response.output.value
            
            # Parse the response to extract object names
            objects = self._parse_objects_from_response(response_text)
            return objects
            
        except Exception as e:
            print(f"Error generating objects: {e}", flush=True)
            return []
    
    def _parse_objects_from_response(self, response_text: str) -> List[str]:
        """Parse object names from the agent's response."""
        objects = []
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', 
                                        '11.', '12.', '13.', '14.', '15.', '16.', '17.', '18.', '19.', '20.'))):
                # Extract object name after the number
                object_name = line.split('.', 1)[1].strip()
                if object_name:
                    objects.append(object_name)
        
        return objects
    
    def generate_objects_and_prompts(self, description: str) -> Tuple[bool, Optional[Dict[str, str]], str]:
        """Generate objects and 2D prompts for the scene objects."""
        try:
            # Get objects for the scene
            objects = self.generate_objects_for_scene(description)
            if not objects:
                return False, None, "No objects generated"
            
            prompts = {}
            for obj in objects:
                prompt = f"Generate visual prompt suitable for 2D image generation for: {obj}"
                response = self.agent.run(prompt, RuleType.PROMPT_GENERATION)
                response_text = response.output.value
                
                # Extract the prompt from the response
                if "Object:" in response_text and "Prompt:" in response_text:
                    prompt_text = response_text.split("Prompt:")[-1].strip()
                    prompts[obj] = prompt_text
                else:
                    prompts[obj] = f"{obj}, detailed 2D image on white background"
            
            return True, prompts, "2D prompts generated successfully"
        except Exception as e:
            return False, None, f"Error generating prompts: {str(e)}"
    
    def clear_memory(self) -> bool:
        """Clear the agent's conversation memory and reset with new randomization."""
        if self.agent:
            self.agent.clear_memory()
            logger.info("Agent memory cleared")
        return True
    
    def move_agent_to_cpu(self):
        """Move the agent's model to CPU to free GPU memory."""
        if self.agent:
            self.agent.move_to_cpu()
    
    def move_agent_to_gpu(self):
        """Move the agent's model back to GPU."""
        if self.agent:
            self.agent.move_to_gpu()
    
    def unload_agent(self):
        """Unload the agent completely to free all memory (GPU and CPU).
        
        Use this in RAM_RESTRICTED mode to save system memory.
        The agent will need to be reloaded before next use.
        """
        self.cleanup()
    
    def cleanup(self):
        """Clean up the agent and free all resources."""
        if self.agent:
            self.agent.cleanup()
            self.agent = None

