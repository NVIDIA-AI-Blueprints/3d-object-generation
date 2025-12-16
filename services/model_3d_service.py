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

"""3D model generation service for converting images to 3D GLB models."""

import os
import base64
import logging
import requests
import datetime
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Default timeout for waiting for Trellis to be ready (in seconds)
TRELLIS_WAIT_TIMEOUT = 300  # 5 minutes

class Model3DService:
    """Service for generating 3D models from images using a REST API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the 3D model service.
        
        Args:
            base_url: Base URL of the 3D model generation service
        """
        self.base_url = base_url.rstrip('/')
        self.infer_endpoint = f"{self.base_url}/v1/infer"
        self.timeout = 300  # 5 minutes timeout for 3D generation
        
    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """Encode an image file to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string with data URI prefix, or None if error
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
                
            # Determine MIME type based on file extension
            file_ext = Path(image_path).suffix.lower()
            mime_types = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.webp': 'image/webp'
            }
            
            mime_type = mime_types.get(file_ext, 'image/png')
            
            # Read and encode the image
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                
            # Create data URI
            data_uri = f"data:{mime_type};base64,{encoded_string}"
            logger.info(f"Successfully encoded image: {image_path}")
            return data_uri
            
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None
    
    def generate_3d_model(self, image_path: str, output_dir: str = "assets/models") -> Tuple[bool, str, Optional[str]]:
        """Generate a 3D model from an image file.
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save the generated GLB file
            
        Returns:
            Tuple of (success, message, glb_file_path)
        """
        try:
            # Wait for Trellis service to be ready before proceeding
            if not self.check_service_health():
                logger.info("Trellis service not ready, waiting...")
                if not self.wait_for_service_ready():
                    return False, "Trellis 3D service not available - timeout waiting for service", None
            
            # Encode image to base64
            base64_image = self.encode_image_to_base64(image_path)
            if not base64_image:
                return False, "Failed to encode image to base64", None
            
            # Make API request
            success, message, glb_data = self.make_infer_request(base64_image)
            if not success:
                return False, message, None
            
            # Save GLB file
            glb_path = self.save_glb_file(glb_data, image_path, output_dir)
            if glb_path:
                return True, f"Successfully generated 3D model", glb_path
            else:
                return False, "Failed to save GLB file", None
                
        except Exception as e:
            logger.error(f"Error in generate_3d_model: {e}")
            return False, f"Error generating 3D model: {str(e)}", None
    
    def make_infer_request(self, base64_image: str) -> Tuple[bool, str, Optional[bytes]]:
        """Make the inference request to the 3D generation API.
        
        Args:
            base64_image: Base64 encoded image with data URI prefix
            
        Returns:
            Tuple of (success, message, glb_binary_data)
            Note: For content filtered responses, success=False and message contains "CONTENT_FILTERED"
        """
        try:
            # Prepare request payload
            payload = {
                "image": base64_image
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            logger.info(f"Making 3D generation request to: {self.infer_endpoint}")
            
            # Make the POST request
            response = requests.post(
                self.infer_endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            # Check response status
            if response.status_code == 200:
                response_data = response.json()
                finish_reason = response_data['artifacts'][0]['finishReason']
                
                if finish_reason == "SUCCESS":
                    # Extract GLB data (adjust key based on actual API response format)
                    glb_base64 = response_data['artifacts'][0]['base64']
         
                    if glb_base64:
                        # Decode base64 to binary
                        glb_binary = base64.b64decode(glb_base64)
                        logger.info("Successfully received 3D model data")
                        return True, "3D model generated successfully", glb_binary
                    else:
                        logger.error(f"No GLB data found in response: {response_data}")
                        return False, "No GLB data in API response", None
                elif finish_reason == "CONTENT_FILTERED":
                    logger.warning("Content filtered by 3D generation service - inappropriate content detected")
                    return False, "CONTENT_FILTERED", None
                else:
                    logger.error(f"API request failed with finish reason: {finish_reason}")
                    return False, f"API request failed: {finish_reason}", None
            else:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return False, f"API request failed: {response.status_code} - {response.text}", None
                
        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            return False, "Request timed out - 3D generation takes time", None
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to 3D generation service at {self.infer_endpoint}")
            return False, "Cannot connect to 3D generation service", None
        except Exception as e:
            logger.error(f"Error making inference request: {e}")
            return False, f"Request error: {str(e)}", None
    
    def save_glb_file(self, glb_data: bytes, original_image_path: str, output_dir: str) -> Optional[str]:
        """Save GLB binary data to a file.
        
        Args:
            glb_data: Binary GLB data
            original_image_path: Path to original image (used for naming)
            output_dir: Directory to save the GLB file
            
        Returns:
            Path to saved GLB file, or None if error
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename based on original image name
            image_name = Path(original_image_path).stem
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            glb_filename = f"{image_name}_{timestamp}.glb"
            glb_path = os.path.join(output_dir, glb_filename)
            
            # Write binary data to file
            with open(glb_path, 'wb') as glb_file:
                glb_file.write(glb_data)
            
            logger.info(f"Saved 3D model to: {glb_path}")
            return glb_path
            
        except Exception as e:
            logger.error(f"Error saving GLB file: {e}")
            return None
    
    def check_service_health(self) -> bool:
        """Check if the 3D generation service is available.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Health endpoint is at /v1/health/ready
            health_url = f"{self.base_url}/v1/health/ready"
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except:
            # If no health endpoint, try the infer endpoint with a HEAD request
            try:
                response = requests.head(self.infer_endpoint, timeout=5)
                return response.status_code in [200, 405]  # 405 is ok, means endpoint exists
            except:
                return False
    
    def wait_for_service_ready(self, timeout: int = TRELLIS_WAIT_TIMEOUT, poll_interval: int = 5) -> bool:
        """Wait for the Trellis 3D generation service to be ready.
        
        Args:
            timeout: Maximum time to wait in seconds (default: 5 minutes)
            poll_interval: Time between health checks in seconds (default: 5)
            
        Returns:
            True if service is ready, False if timeout reached
        """
        start_time = time.time()
        logger.info(f"Waiting for Trellis 3D service to be ready (timeout: {timeout}s)...")
        
        while time.time() - start_time < timeout:
            if self.check_service_health():
                elapsed = time.time() - start_time
                logger.info(f"Trellis 3D service is ready (waited {elapsed:.1f}s)")
                return True
            
            elapsed = time.time() - start_time
            logger.info(f"Trellis not ready yet, waiting... ({elapsed:.0f}s / {timeout}s)")
            time.sleep(poll_interval)
        
        logger.error(f"Trellis 3D service not ready after {timeout}s timeout")
        return False
    
    def batch_generate_models(self, image_paths: list, output_dir: str = "assets/models") -> Dict[str, Any]:
        """Generate 3D models for multiple images.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save generated GLB files
            
        Returns:
            Dictionary with results for each image
        """
        results = {
            "successful": [],
            "failed": [],
            "total": len(image_paths)
        }
        
        for image_path in image_paths:
            logger.info(f"Processing: {image_path}")
            success, message, glb_path = self.generate_3d_model(image_path, output_dir)
            
            if success:
                results["successful"].append({
                    "image_path": image_path,
                    "glb_path": glb_path,
                    "message": message
                })
            else:
                results["failed"].append({
                    "image_path": image_path,
                    "error": message
                })
        
        logger.info(f"Batch processing complete: {len(results['successful'])}/{results['total']} successful")
        return results


def create_sample_request():
    """Create a sample Python request demonstrating the service usage."""
    
    # Example usage
    model_service = Model3DService(base_url="http://localhost:8000")
    
    # Example 1: Single image generation
    image_path = "bookshelf_42_20250806_192025.png"
    success, message, glb_path = model_service.generate_3d_model(image_path)
    
    if success:
        print(f"✅ Success: {message}")
        print(f"📁 GLB file saved to: {glb_path}")
    elif message == "CONTENT_FILTERED":
        print(f"🚫 Content filtered: Image contains inappropriate content")
    else:
        print(f"❌ Failed: {message}")
    
    # Example 2: Check service health
    if model_service.check_service_health():
        print("✅ 3D generation service is healthy")
    else:
        print("❌ 3D generation service is not available")
    
    # # Example 3: Batch processing
    # image_list = [
    #     "bookshelf_42_20250806_192025.png",
    #     "another_image.jpg"
    # ]
    # batch_results = model_service.batch_generate_models(image_list)
    # print(f"📊 Batch results: {batch_results['total']} images processed")


if __name__ == "__main__":
    # Run the sample request
    create_sample_request() 