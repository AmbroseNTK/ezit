# Standard library imports
import json
import re
import os
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING

# import cv2

import numpy as np
from PIL import Image
import openai
from sklearn.cluster import KMeans

class Node:

    """Represents a processing node in the workflow"""
    
    def __init__(self, node_id: str, node_type: str, parameters: Optional[Dict[str, Any]] = None, 
                 inputs: Optional[List[str]] = None, outputs: Optional[List[str]] = None):
        self.node_id = node_id
        self.node_type = node_type
        self.parameters = parameters or {}
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.result = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "parameters": self.parameters,
            "inputs": self.inputs,
            "outputs": self.outputs
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Create node from dictionary representation"""
        return cls(
            node_id=data["node_id"],
            node_type=data["node_type"],
            parameters=data.get("parameters", {}),
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", [])
        )

class WorkflowProcessor:
    """Main workflow processor for image processing"""
    
    def __init__(self):
        self.nodes = {}
        self.connections = []
        self.node_specs = self._load_node_specifications()
        self.execution_order = []
        
    def _load_node_specifications(self) -> Dict[str, Any]:
        """Load node specifications from node_spec.txt"""
        specs = {}
        
        with open('node_spec.txt', 'r') as f:
            content = f.read()
            
        # Parse the specifications using regex
        node_blocks = re.split(r'### \d+\. ', content)[1:]  # Skip the header
        
        for block in node_blocks:
            lines = block.strip().split('\n')
            # Extract node name from the first line - look for the name field
            node_name = None
            for line in lines:
                if line.strip().startswith('- **name**: '):
                    node_name = line.split('"')[1]  # Extract name between quotes
                    break
            
            if not node_name:
                continue  # Skip if no name found
                
            spec = {
                "name": node_name,
                "description": "",
                "parameters": {},
                "inputs": [],
                "outputs": [],
                "operator": ""
            }
            
            for line in lines[1:]:
                line = line.strip()
                if line.startswith("- **description**: "):
                    spec["description"] = line.split(": ", 1)[1].strip('"')
                elif line.startswith("- **parameters**: "):
                    # Parse parameters (simplified parsing)
                    param_text = line.split(": ", 1)[1]
                    if param_text != "{}":
                        # Extract parameter definitions
                        spec["parameters"] = self._parse_parameters(param_text)
                elif line.startswith("- **inputs**: "):
                    inputs_text = line.split(": ", 1)[1]
                    if inputs_text != "[]":
                        spec["inputs"] = [s.strip('"') for s in inputs_text.strip('[]').split(',')]
                elif line.startswith("- **outputs**: "):
                    outputs_text = line.split(": ", 1)[1]
                    if outputs_text != "[]":
                        spec["outputs"] = [s.strip('"') for s in outputs_text.strip('[]').split(',')]
                elif line.startswith("- **operator**: "):
                    spec["operator"] = line.split(": ", 1)[1].strip('"')
            
            specs[node_name] = spec
            
        # Debug output
        print(f"Loaded node specifications: {list(specs.keys())}")
            
        return specs
    
    def _parse_parameters(self, param_text: str) -> Dict[str, Any]:
        """Parse parameters from text representation"""
        params = {}
        # This is a simplified parser - in production you'd want a more robust solution
        try:
            # Extract parameter definitions using regex
            param_matches = re.findall(r'"([^"]+)":\s*{([^}]+)}', param_text)
            for param_name, param_def in param_matches:
                type_match = re.search(r'"type":\s*"([^"]+)"', param_def)
                default_match = re.search(r'"default":\s*([^,}]+)', param_def)
                desc_match = re.search(r'"description":\s*"([^"]+)"', param_def)
                
                if type_match:
                    param_type = type_match.group(1)
                    default_val = None
                    if default_match:
                        default_str = default_match.group(1).strip()
                        if default_str.startswith('[') and default_str.endswith(']'):
                            # Parse tuple/list
                            default_val = [int(x.strip()) for x in default_str.strip('[]').split(',')]
                        elif default_str.isdigit():
                            default_val = int(default_str)
                        elif default_str.replace('.', '').replace('-', '').isdigit():
                            default_val = float(default_str)
                        elif default_str.startswith('"') and default_str.endswith('"'):
                            default_val = default_str.strip('"')
                        else:
                            default_val = default_str
                    
                    params[param_name] = {
                        "type": param_type,
                        "default": default_val,
                        "description": desc_match.group(1) if desc_match else ""
                    }
        except Exception as e:
            print(f"Warning: Could not parse parameters: {e}")
            
        return params
    
    def add_node(self, node: Node) -> None:
        """Add a node to the workflow"""
        self.nodes[node.node_id] = node
        
    def add_connection(self, from_node: str, to_node: str, from_output: str = "image", to_input: str = "image") -> None:
        """Add a connection between nodes"""
        self.connections.append({
            "from_node": from_node,
            "to_node": to_node,
            "from_output": from_output,
            "to_input": to_input
        })
        
    def _determine_execution_order(self) -> List[str]:
        """Determine the topological order of node execution"""
        # Build adjacency list
        graph = {node_id: [] for node_id in self.nodes}
        in_degree = {node_id: 0 for node_id in self.nodes}
        
        for conn in self.connections:
            graph[conn["from_node"]].append(conn["to_node"])
            in_degree[conn["to_node"]] += 1
            
        # Topological sort using Kahn's algorithm
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            current = queue.pop(0)
            order.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        if len(order) != len(self.nodes):
            raise ValueError("Workflow contains cycles or disconnected nodes")
            
        return order
    
    def _execute_node(self, node: Node, input_data: Dict[str, Any]) -> Dict[str, Any]:
        import cv2
        """Execute a single node with given input data"""
        # type: ignore  # Suppress cv2-related linter warnings
        operator = self.node_specs[node.node_type]["operator"]
        
        print(f"Executing node {node.node_id} of type {node.node_type} with operator {operator}")
        
        if operator == "input":
            return {"image": input_data.get("image")}
            
        elif operator == "output":
            return {"image": input_data.get("image")}
            
        elif operator == "gaussian_blur":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for gaussian_blur operation")
            kernel_size = tuple(node.parameters.get("kernel_size", [5, 5]))
            sigma_x = node.parameters.get("sigma_x", 0.0)
            sigma_y = node.parameters.get("sigma_y", 0.0)
            result = cv2.GaussianBlur(image, kernel_size, sigma_x, sigma_y)
            return {"image": result}
            
        elif operator == "median_blur":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for median_blur operation")
            kernel_size = node.parameters.get("kernel_size", 5)
            result = cv2.medianBlur(image, kernel_size)
            return {"image": result}
            
        elif operator == "bilateral_filter":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for bilateral_filter operation")
            d = node.parameters.get("d", 15)
            sigma_color = node.parameters.get("sigma_color", 75.0)
            sigma_space = node.parameters.get("sigma_space", 75.0)
            result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            return {"image": result}
            
        elif operator == "rgb2gray":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for rgb2gray operation")
            if len(image.shape) == 3:
                result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                result = image
            return {"image": result}
            
        elif operator == "gray2rgb":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for gray2rgb operation")
            if len(image.shape) == 2:
                result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                result = image
            return {"image": result}
            
        elif operator == "canny_edge":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for canny_edge operation")
            threshold1 = node.parameters.get("threshold1", 100.0)
            threshold2 = node.parameters.get("threshold2", 200.0)
            aperture_size = node.parameters.get("aperture_size", 3)
            result = cv2.Canny(image, threshold1, threshold2, apertureSize=aperture_size)
            return {"image": result}
            
        elif operator == "sobel_edge":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for sobel_edge operation")
            dx = node.parameters.get("dx", 1)
            dy = node.parameters.get("dy", 1)
            ksize = node.parameters.get("ksize", 3)
            scale = node.parameters.get("scale", 1.0)
            delta = node.parameters.get("delta", 0.0)
            result = cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize=ksize, scale=scale, delta=delta)
            result = np.uint8(np.absolute(result))
            return {"image": result}
            
        elif operator == "threshold":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for threshold operation")
            thresh = node.parameters.get("thresh", 127.0)
            maxval = node.parameters.get("maxval", 255.0)
            thresh_type = node.parameters.get("type", "THRESH_BINARY")
            
            # Convert string to cv2 constant
            if thresh_type == "THRESH_BINARY":
                thresh_type = cv2.THRESH_BINARY  # type: ignore
            elif thresh_type == "THRESH_BINARY_INV":
                thresh_type = cv2.THRESH_BINARY_INV  # type: ignore
            elif thresh_type == "THRESH_TRUNC":
                thresh_type = cv2.THRESH_TRUNC  # type: ignore
            elif thresh_type == "THRESH_TOZERO":
                thresh_type = cv2.THRESH_TOZERO  # type: ignore
            elif thresh_type == "THRESH_TOZERO_INV":
                thresh_type = cv2.THRESH_TOZERO_INV  # type: ignore
                
            _, result = cv2.threshold(image, thresh, maxval, thresh_type)  # type: ignore
            return {"image": result}
            
        elif operator == "adaptive_threshold":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for adaptive_threshold operation")
            max_value = node.parameters.get("max_value", 255.0)
            adaptive_method = node.parameters.get("adaptive_method", "ADAPTIVE_THRESH_GAUSSIAN_C")
            threshold_type = node.parameters.get("threshold_type", "THRESH_BINARY")
            block_size = node.parameters.get("block_size", 11)
            c_value = node.parameters.get("c", 2.0)
            
            # Convert strings to cv2 constants
            if adaptive_method == "ADAPTIVE_THRESH_GAUSSIAN_C":
                adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C  # type: ignore
            elif adaptive_method == "ADAPTIVE_THRESH_MEAN_C":
                adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C  # type: ignore
                
            if threshold_type == "THRESH_BINARY":
                threshold_type = cv2.THRESH_BINARY  # type: ignore
            elif threshold_type == "THRESH_BINARY_INV":
                threshold_type = cv2.THRESH_BINARY_INV  # type: ignore
                
            result = cv2.adaptiveThreshold(image, max_value, adaptive_method, threshold_type, block_size, c_value)  # type: ignore
            return {"image": result}
            
        elif operator == "morphology":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for morphology operation")
            operation = node.parameters.get("operation", "MORPH_OPEN")
            kernel_size = node.parameters.get("kernel_size", 3)
            iterations = node.parameters.get("iterations", 1)
            
            # Create kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))  # type: ignore
            
            # Convert string to cv2 constant
            if operation == "MORPH_OPEN":
                operation = cv2.MORPH_OPEN  # type: ignore
            elif operation == "MORPH_CLOSE":
                operation = cv2.MORPH_CLOSE  # type: ignore
            elif operation == "MORPH_ERODE":
                operation = cv2.MORPH_ERODE  # type: ignore
            elif operation == "MORPH_DILATE":
                operation = cv2.MORPH_DILATE  # type: ignore
                
            result = cv2.morphologyEx(image, operation, kernel, iterations=iterations)  # type: ignore
            return {"image": result}
            
        elif operator == "histogram_equalization":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for histogram_equalization operation")
            if len(image.shape) == 2:
                result = cv2.equalizeHist(image)
            else:
                # For color images, convert to YUV and equalize Y channel
                yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            return {"image": result}
            
        elif operator == "clahe":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for clahe operation")
            clip_limit = node.parameters.get("clip_limit", 2.0)
            tile_grid_size = tuple(node.parameters.get("tile_grid_size", [8, 8]))
            
            if len(image.shape) == 2:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                result = clahe.apply(image)
            else:
                # For color images, apply to each channel
                yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                yuv[:,:,0] = clahe.apply(yuv[:,:,0])
                result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            return {"image": result}
            
        elif operator == "color_space_convert":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for color_space_convert operation")
            code = node.parameters.get("code", "COLOR_BGR2HSV")
            
            # Convert string to cv2 constant
            if code == "COLOR_BGR2HSV":
                code = cv2.COLOR_BGR2HSV
            elif code == "COLOR_HSV2BGR":
                code = cv2.COLOR_HSV2BGR
            elif code == "COLOR_BGR2GRAY":
                code = cv2.COLOR_BGR2GRAY
            elif code == "COLOR_GRAY2BGR":
                code = cv2.COLOR_GRAY2BGR
            elif code == "COLOR_BGR2LAB":
                code = cv2.COLOR_BGR2LAB
            elif code == "COLOR_LAB2BGR":
                code = cv2.COLOR_LAB2BGR
                
            result = cv2.cvtColor(image, code)
            return {"image": result}
            
        elif operator == "resize":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for resize operation")
            width = node.parameters.get("width", 640)
            height = node.parameters.get("height", 480)
            interpolation = node.parameters.get("interpolation", "INTER_LINEAR")
            
            # Convert string to cv2 constant
            if interpolation == "INTER_LINEAR":
                interpolation = cv2.INTER_LINEAR
            elif interpolation == "INTER_NEAREST":
                interpolation = cv2.INTER_NEAREST
            elif interpolation == "INTER_CUBIC":
                interpolation = cv2.INTER_CUBIC
            elif interpolation == "INTER_AREA":
                interpolation = cv2.INTER_AREA
                
            result = cv2.resize(image, (width, height), interpolation=interpolation)
            return {"image": result}
            
        elif operator == "crop":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for crop operation")
            x = node.parameters.get("x", 0)
            y = node.parameters.get("y", 0)
            width = node.parameters.get("width", 100)
            height = node.parameters.get("height", 100)
            
            result = image[y:y+height, x:x+width]
            return {"image": result}
            
        elif operator == "brightness_contrast":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for brightness_contrast operation")
            alpha = node.parameters.get("alpha", 1.0)  # Contrast
            beta = node.parameters.get("beta", 0.0)    # Brightness
            
            result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            return {"image": result}
            
        elif operator == "gamma_correction":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for gamma_correction operation")
            gamma = node.parameters.get("gamma", 1.0)
            
            # Build lookup table
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            result = cv2.LUT(image, table)
            return {"image": result}
            
        elif operator == "add_noise":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for add_noise operation")
            mean = node.parameters.get("mean", 0.0)
            std = node.parameters.get("std", 25.0)
            
            noise = np.random.normal(mean, std, image.shape).astype(np.float32)
            result = cv2.add(image.astype(np.float32), noise)
            result = np.clip(result, 0, 255).astype(np.uint8)
            return {"image": result}
            
        elif operator == "add_text":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for add_text operation")
            
            # Get text parameters
            text = node.parameters.get("text", "Hello World")
            x = node.parameters.get("x", 10)
            y = node.parameters.get("y", 30)
            font_scale = node.parameters.get("font_scale", 1.0)
            color = tuple(node.parameters.get("color", [255, 255, 255]))  # BGR format
            thickness = node.parameters.get("thickness", 2)
            font_name = node.parameters.get("font", "FONT_HERSHEY_SIMPLEX")
            
            # Convert string to cv2 font constant
            if font_name == "FONT_HERSHEY_SIMPLEX":
                font = cv2.FONT_HERSHEY_SIMPLEX
            elif font_name == "FONT_HERSHEY_PLAIN":
                font = cv2.FONT_HERSHEY_PLAIN
            elif font_name == "FONT_HERSHEY_DUPLEX":
                font = cv2.FONT_HERSHEY_DUPLEX
            elif font_name == "FONT_HERSHEY_COMPLEX":
                font = cv2.FONT_HERSHEY_COMPLEX
            elif font_name == "FONT_HERSHEY_TRIPLEX":
                font = cv2.FONT_HERSHEY_TRIPLEX
            elif font_name == "FONT_HERSHEY_COMPLEX_SMALL":
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            elif font_name == "FONT_HERSHEY_SCRIPT_SIMPLEX":
                font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
            elif font_name == "FONT_HERSHEY_SCRIPT_COMPLEX":
                font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX  # Default
            
            # Create a copy of the image to avoid modifying the original
            result = image.copy()
            
            # Add text to the image
            cv2.putText(result, text, (x, y), font, font_scale, color, thickness)
            
            return {"image": result}
            
        elif operator == "unsharp_mask":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for unsharp_mask operation")
            kernel_size = tuple(node.parameters.get("kernel_size", [5, 5]))
            sigma = node.parameters.get("sigma", 1.0)
            amount = node.parameters.get("amount", 1.0)
            threshold = node.parameters.get("threshold", 0.0)
            blurred = cv2.GaussianBlur(image, kernel_size, sigma)
            sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
            if threshold > 0:
                low_contrast_mask = np.abs(image - blurred) < threshold
                np.copyto(sharpened, image, where=low_contrast_mask)
            return {"image": sharpened}

        elif operator == "laplacian_sharpen":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for laplacian_sharpen operation")
            ksize = node.parameters.get("ksize", 3)
            scale = node.parameters.get("scale", 1.0)
            delta = node.parameters.get("delta", 0.0)
            lap = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize, scale=scale, delta=delta)
            sharp = cv2.convertScaleAbs(image - lap)
            return {"image": sharp}

        elif operator == "emboss":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for emboss operation")
            kernel_size = node.parameters.get("kernel_size", 3)
            strength = node.parameters.get("strength", 1.0)
            kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]) * strength
            result = cv2.filter2D(image, -1, kernel)
            return {"image": result}

        elif operator == "sepia":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for sepia operation")
            intensity = node.parameters.get("intensity", 0.8)
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            sepia_img = cv2.transform(image, sepia_filter)
            sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
            result = cv2.addWeighted(image, 1 - intensity, sepia_img, intensity, 0)
            return {"image": result}

        elif operator == "vintage":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for vintage operation")
            warmth = node.parameters.get("warmth", 0.1)
            saturation = node.parameters.get("saturation", 0.8)
            contrast = node.parameters.get("contrast", 1.2)
            img = image.astype(np.float32)
            img = img * contrast
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            img[..., 1] *= saturation
            img[..., 0] += warmth * 10
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            return {"image": img}

        elif operator == "cartoon":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for cartoon operation")
            edge_strength = node.parameters.get("edge_strength", 0.5)
            color_levels = node.parameters.get("color_levels", 8)
            blur_strength = node.parameters.get("blur_strength", 7)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.medianBlur(gray, 7)
            edges = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
            data = np.float32(image.reshape((-1, 3)))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
            kmeans = KMeans(n_clusters=color_levels, n_init='auto', max_iter=20)
            labels = kmeans.fit_predict(data)
            quant = kmeans.cluster_centers_[labels.flatten()].reshape(image.shape)
            quant = np.clip(quant, 0, 255).astype(np.uint8)
            blurred = cv2.bilateralFilter(quant, d=blur_strength, sigmaColor=200, sigmaSpace=200)
            if edges.dtype != np.uint8:
                edges = edges.astype(np.uint8)
            cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
            return {"image": cartoon}

        elif operator == "oil_painting":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for oil_painting operation")
            radius = node.parameters.get("radius", 4)
            intensity = node.parameters.get("intensity", 30)
            result = None
            try:
                # import cv2.xphoto
                
                result = cv2.xphoto.oilPainting(image, radius, intensity)
            except Exception:
                result = None
            if result is None:
                # Fallback: use median blur and quantization
                result = cv2.medianBlur(image, radius)
            return {"image": result}

        elif operator == "watercolor":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for watercolor operation")
            blur_radius = node.parameters.get("blur_radius", 5)
            edge_strength = node.parameters.get("edge_strength", 0.3)
            saturation = node.parameters.get("saturation", 1.2)
            blurred = cv2.edgePreservingFilter(image, flags=1, sigma_s=60, sigma_r=0.4)
            img = cv2.bilateralFilter(blurred, d=9, sigmaColor=75, sigmaSpace=75)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] *= saturation
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return {"image": img}

        elif operator == "sketch":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for sketch operation")
            blur_strength = node.parameters.get("blur_strength", 5)
            edge_threshold = node.parameters.get("edge_threshold", 50.0)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            inv = 255 - gray
            blur = cv2.GaussianBlur(inv, (blur_strength | 1, blur_strength | 1), 0)
            sketch = cv2.divide(gray, 255 - blur, scale=256)
            _, sketch = cv2.threshold(sketch, edge_threshold, 255, cv2.THRESH_BINARY)
            return {"image": cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)}

        elif operator == "invert":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for invert operation")
            result = cv2.bitwise_not(image)
            return {"image": result}

        elif operator == "solarize":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for solarize operation")
            threshold = node.parameters.get("threshold", 128)
            result = np.where(image < threshold, image, 255 - image)
            result = result.astype(np.uint8)
            return {"image": result}

        elif operator == "posterize":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for posterize operation")
            levels = node.parameters.get("levels", 4)
            shift = 8 - int(np.log2(levels))
            result = np.right_shift(image, shift)
            result = np.left_shift(result, shift)
            return {"image": result}

        elif operator == "color_balance":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for color_balance operation")
            shadows = np.array(node.parameters.get("shadows", [1.0, 1.0, 1.0]))
            midtones = np.array(node.parameters.get("midtones", [1.0, 1.0, 1.0]))
            highlights = np.array(node.parameters.get("highlights", [1.0, 1.0, 1.0]))
            img = image.astype(np.float32)
            mask_shadows = img < 85
            mask_highlights = img > 170
            mask_midtones = (~mask_shadows) & (~mask_highlights)
            img[mask_shadows] *= shadows
            img[mask_midtones] *= midtones
            img[mask_highlights] *= highlights
            img = np.clip(img, 0, 255).astype(np.uint8)
            return {"image": img}

        elif operator == "color_temperature":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for color_temperature operation")
            temperature = node.parameters.get("temperature", 0.0)
            img = image.astype(np.float32)
            if temperature > 0:
                img[..., 2] += temperature
            else:
                img[..., 0] -= temperature
            img = np.clip(img, 0, 255).astype(np.uint8)
            return {"image": img}

        elif operator == "tint":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for tint operation")
            tint_color = np.array(node.parameters.get("tint_color", [255, 200, 150]))
            strength = node.parameters.get("strength", 0.3)
            img = image.astype(np.float32)
            img = (1 - strength) * img + strength * tint_color
            img = np.clip(img, 0, 255).astype(np.uint8)
            return {"image": img}

        elif operator == "duotone":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for duotone operation")
            color1 = np.array(node.parameters.get("color1", [0, 0, 255]))
            color2 = np.array(node.parameters.get("color2", [255, 255, 255]))
            blend = node.parameters.get("blend", 0.5)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            norm = gray / 255.0
            img = (1 - norm[..., None]) * color1 + norm[..., None] * color2
            img = np.clip(img, 0, 255).astype(np.uint8)
            return {"image": img}

        elif operator == "gradient_map":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for gradient_map operation")
            gradient_colors = node.parameters.get("gradient_colors", [[0, 0, 0], [255, 255, 255]])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            norm = gray / 255.0
            gradient_colors = np.array(gradient_colors)
            idx = (norm * (len(gradient_colors) - 1)).astype(int)
            img = gradient_colors[idx]
            img = np.clip(img, 0, 255).astype(np.uint8)
            return {"image": img}

        elif operator == "lens_blur":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for lens_blur operation")
            radius = node.parameters.get("radius", 10)
            center_x = node.parameters.get("center_x", 0.5)
            center_y = node.parameters.get("center_y", 0.5)
            h, w = image.shape[:2]
            Y, X = np.ogrid[:h, :w]
            cx, cy = int(center_x * w), int(center_y * h)
            mask = ((X - cx) ** 2 + (Y - cy) ** 2) > (radius ** 2)
            blurred = cv2.GaussianBlur(image, (radius | 1, radius | 1), 0)
            result = image.copy()
            result[mask] = blurred[mask]
            return {"image": result}

        elif operator == "motion_blur":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for motion_blur operation")
            angle = node.parameters.get("angle", 0.0)
            strength = node.parameters.get("strength", 15)
            kernel = np.zeros((strength, strength))
            kernel[int((strength - 1) / 2), :] = np.ones(strength)
            kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((strength / 2 - 0.5, strength / 2 - 0.5), angle, 1.0), (strength, strength))
            kernel = kernel / np.sum(kernel)
            result = cv2.filter2D(image, -1, kernel)
            return {"image": result}

        elif operator == "radial_blur":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for radial_blur operation")
            center_x = node.parameters.get("center_x", 0.5)
            center_y = node.parameters.get("center_y", 0.5)
            strength = node.parameters.get("strength", 0.1)
            h, w = image.shape[:2]
            Y, X = np.ogrid[:h, :w]
            cx, cy = int(center_x * w), int(center_y * h)
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            max_dist = np.max(dist)
            blur_amount = (dist / max_dist) * strength * 20
            result = image.copy()
            for i in range(1, int(np.max(blur_amount)) + 1):
                mask = (blur_amount >= i - 1) & (blur_amount < i)
                if np.any(mask):
                    blurred = cv2.GaussianBlur(image, (i * 2 + 1, i * 2 + 1), 0)
                    result[mask] = blurred[mask]
            return {"image": result}

        elif operator == "perspective_transform":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for perspective_transform operation")
            src_points = np.array(node.parameters.get("src_points", [[0, 0], [640, 0], [640, 480], [0, 480]]), dtype=np.float32)
            dst_points = np.array(node.parameters.get("dst_points", [[50, 50], [590, 30], [590, 450], [50, 430]]), dtype=np.float32)
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            h, w = image.shape[:2]
            result = cv2.warpPerspective(image, M, (w, h))
            return {"image": result}

        elif operator == "affine_transform":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for affine_transform operation")
            angle = node.parameters.get("angle", 0.0)
            scale = node.parameters.get("scale", 1.0)
            tx = node.parameters.get("tx", 0.0)
            ty = node.parameters.get("ty", 0.0)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            result = cv2.warpAffine(image, M, (w, h))
            return {"image": result}

        elif operator == "mirror":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for mirror operation")
            direction = node.parameters.get("direction", "horizontal")
            if direction == "horizontal":
                result = cv2.flip(image, 1)
            elif direction == "vertical":
                result = cv2.flip(image, 0)
            elif direction == "both":
                result = cv2.flip(image, -1)
            else:
                result = image
            return {"image": result}

        elif operator == "rotate":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for rotate operation")
            angle = node.parameters.get("angle", 90.0)
            center_x = node.parameters.get("center_x", -1)
            center_y = node.parameters.get("center_y", -1)
            scale = node.parameters.get("scale", 1.0)
            h, w = image.shape[:2]
            if center_x == -1 or center_y == -1:
                center = (w // 2, h // 2)
            else:
                center = (int(center_x), int(center_y))
            M = cv2.getRotationMatrix2D(center, angle, scale)
            result = cv2.warpAffine(image, M, (w, h))
            return {"image": result}

        elif operator == "tilt_shift":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for tilt_shift operation")
            focus_center = node.parameters.get("focus_center", 0.5)
            focus_width = node.parameters.get("focus_width", 0.2)
            blur_strength = node.parameters.get("blur_strength", 15)
            h, w = image.shape[:2]
            mask = np.zeros((h, w), np.float32)
            y1 = int((focus_center - focus_width / 2) * h)
            y2 = int((focus_center + focus_width / 2) * h)
            mask[y1:y2, :] = 1.0
            mask = cv2.GaussianBlur(mask, (blur_strength | 1, blur_strength | 1), 0)
            blurred = cv2.GaussianBlur(image, (blur_strength | 1, blur_strength | 1), 0)
            result = image * mask[..., None] + blurred * (1 - mask[..., None])
            result = np.clip(result, 0, 255).astype(np.uint8)
            return {"image": result}

        elif operator == "fisheye":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for fisheye operation")
            strength = node.parameters.get("strength", 0.3)
            center_x = node.parameters.get("center_x", 0.5)
            center_y = node.parameters.get("center_y", 0.5)
            h, w = image.shape[:2]
            K = np.array([[w, 0, w * center_x], [0, w, h * center_y], [0, 0, 1]], dtype=np.float32)
            D = np.array([strength, 0, 0, 0], dtype=np.float32)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)
            result = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            return {"image": result}

        elif operator == "barrel_distortion":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for barrel_distortion operation")
            k1 = node.parameters.get("k1", 0.1)
            k2 = node.parameters.get("k2", 0.05)
            k3 = node.parameters.get("k3", 0.0)
            h, w = image.shape[:2]
            fx = w
            fy = h
            cx = w / 2
            cy = h / 2
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            D = np.array([k1, k2, k3, 0], dtype=np.float32)
            map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)
            result = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
            return {"image": result}

        elif operator == "pincushion_distortion":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for pincushion_distortion operation")
            k1 = node.parameters.get("k1", -0.1)
            k2 = node.parameters.get("k2", -0.05)
            k3 = node.parameters.get("k3", 0.0)
            h, w = image.shape[:2]
            fx = w
            fy = h
            cx = w / 2
            cy = h / 2
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            D = np.array([k1, k2, k3, 0], dtype=np.float32)
            map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)
            result = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
            return {"image": result}

        elif operator == "wave_distortion":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for wave_distortion operation")
            amplitude = node.parameters.get("amplitude", 10.0)
            frequency = node.parameters.get("frequency", 0.1)
            phase = node.parameters.get("phase", 0.0)
            h, w = image.shape[:2]
            map_y, map_x = np.indices((h, w), dtype=np.float32)
            map_x_new = map_x + amplitude * np.sin(2 * np.pi * map_y * frequency + phase)
            # Ensure map_x_new and map_y are contiguous and of type float32
            map_x_new = np.ascontiguousarray(map_x_new, dtype=np.float32)
            map_y = np.ascontiguousarray(map_y, dtype=np.float32)
            result = cv2.remap(image, map_y, map_x_new, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            return {"image": result}

        elif operator == "ripple_effect":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for ripple_effect operation")
            center_x = node.parameters.get("center_x", 0.5)
            center_y = node.parameters.get("center_y", 0.5)
            amplitude = node.parameters.get("amplitude", 20.0)
            frequency = node.parameters.get("frequency", 0.05)
            h, w = image.shape[:2]
            cx, cy = int(center_x * w), int(center_y * h)
            Y, X = np.indices((h, w), dtype=np.float32)
            r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            map_x = X + amplitude * np.sin(2 * np.pi * r * frequency)
            map_y = Y + amplitude * np.cos(2 * np.pi * r * frequency)
            result = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            return {"image": result}

        elif operator == "pixelate":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for pixelate operation")
            block_size = node.parameters.get("block_size", 10)
            h, w = image.shape[:2]
            temp = cv2.resize(image, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
            result = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
            return {"image": result}

        elif operator == "mosaic":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for mosaic operation")
            tile_size = node.parameters.get("tile_size", 20)
            color_variation = node.parameters.get("color_variation", 0.3)
            h, w = image.shape[:2]
            result = image.copy()
            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    roi = image[y:y+tile_size, x:x+tile_size]
                    color = np.mean(roi.reshape(-1, 3), axis=0)
                    color += np.random.uniform(-color_variation*128, color_variation*128, 3)
                    color = np.clip(color, 0, 255)
                    result[y:y+tile_size, x:x+tile_size] = color
            return {"image": result.astype(np.uint8)}

        elif operator == "halftone":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for halftone operation")
            dot_size = node.parameters.get("dot_size", 5)
            angle = node.parameters.get("angle", 45.0)
            h, w = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result = np.ones_like(image) * 255
            for y in range(0, h, dot_size):
                for x in range(0, w, dot_size):
                    roi = gray[y:y+dot_size, x:x+dot_size]
                    avg = np.mean(roi)
                    radius = int((1 - avg / 255.0) * (dot_size / 2))
                    cv2.circle(result, (x + dot_size // 2, y + dot_size // 2), radius, (0, 0, 0), -1)
            return {"image": result}

        elif operator == "crosshatch":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for crosshatch operation")
            line_spacing = node.parameters.get("line_spacing", 10)
            line_thickness = node.parameters.get("line_thickness", 1)
            h, w = image.shape[:2]
            result = image.copy()
            for y in range(0, h, line_spacing):
                cv2.line(result, (0, y), (w, y), (0, 0, 0), line_thickness)
            for x in range(0, w, line_spacing):
                cv2.line(result, (x, 0), (x, h), (0, 0, 0), line_thickness)
            return {"image": result}

        elif operator == "stipple":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for stipple operation")
            dot_size = node.parameters.get("dot_size", 2)
            dot_spacing = node.parameters.get("dot_spacing", 5)
            h, w = image.shape[:2]
            result = image.copy()
            for y in range(0, h, dot_spacing):
                for x in range(0, w, dot_spacing):
                    if np.random.rand() > 0.5:
                        cv2.circle(result, (x, y), dot_size, (0, 0, 0), -1)
            return {"image": result}

        elif operator == "engrave":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for engrave operation")
            angle = node.parameters.get("angle", 45.0)
            depth = node.parameters.get("depth", 0.5)
            kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) * depth
            result = cv2.filter2D(image, -1, kernel)
            return {"image": result}

        elif operator == "emboss_3d":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for emboss_3d operation")
            depth = node.parameters.get("depth", 0.5)
            light_angle = node.parameters.get("light_angle", 45.0)
            kernel = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]) * depth
            result = cv2.filter2D(image, -1, kernel)
            return {"image": result}

        elif operator == "relief":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for relief operation")
            strength = node.parameters.get("strength", 0.5)
            angle = node.parameters.get("angle", 45.0)
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) * strength
            result = cv2.filter2D(image, -1, kernel)
            return {"image": result}

        elif operator == "chrome":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for chrome operation")
            reflection_strength = node.parameters.get("reflection_strength", 0.7)
            metallic_quality = node.parameters.get("metallic_quality", 0.8)
            img = cv2.GaussianBlur(image, (7, 7), 0)
            img = cv2.addWeighted(image, 1 - metallic_quality, img, metallic_quality, 0)
            img = cv2.addWeighted(img, 1, np.full_like(img, 200), reflection_strength, 0)
            img = np.clip(img, 0, 255).astype(np.uint8)
            return {"image": img}

        elif operator == "glass":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for glass operation")
            transparency = node.parameters.get("transparency", 0.3)
            refraction = node.parameters.get("refraction", 0.1)
            h, w = image.shape[:2]
            result = image.copy().astype(np.float32)
            for y in range(h):
                for x in range(w):
                    dx = int(refraction * np.sin(2 * np.pi * y / 60))
                    dy = int(refraction * np.cos(2 * np.pi * x / 60))
                    sx = np.clip(x + dx, 0, w - 1)
                    sy = np.clip(y + dy, 0, h - 1)
                    result[y, x] = (1 - transparency) * image[y, x] + transparency * image[sy, sx]
            result = np.clip(result, 0, 255).astype(np.uint8)
            return {"image": result}

        elif operator == "neon_glow":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for neon_glow operation")
            glow_color = np.array(node.parameters.get("glow_color", [0, 255, 255]))
            glow_strength = node.parameters.get("glow_strength", 0.8)
            glow_radius = node.parameters.get("glow_radius", 10)
            edges = cv2.Canny(image, 100, 200)
            edges_colored = np.zeros_like(image)
            for c in range(3):
                edges_colored[..., c] = edges * (glow_color[c] / 255.0)
            blurred = cv2.GaussianBlur(edges_colored, (glow_radius | 1, glow_radius | 1), 0)
            result = cv2.addWeighted(image, 1.0, blurred, glow_strength, 0)
            return {"image": result}

        elif operator == "fire_effect":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for fire_effect operation")
            intensity = node.parameters.get("intensity", 0.7)
            color_temperature = node.parameters.get("color_temperature", 0.8)
            fire = np.zeros_like(image)
            fire[..., 2] = np.clip(image[..., 2] * (1 + intensity), 0, 255)
            fire[..., 1] = np.clip(image[..., 1] * (1 - color_temperature), 0, 255)
            result = cv2.addWeighted(image, 1 - intensity, fire, intensity, 0)
            return {"image": result}

        elif operator == "smoke_effect":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for smoke_effect operation")
            density = node.parameters.get("density", 0.5)
            opacity = node.parameters.get("opacity", 0.3)
            h, w = image.shape[:2]
            smoke = np.random.normal(128, 50, (h, w)).astype(np.uint8)
            smoke = cv2.GaussianBlur(smoke, (31, 31), 0)
            smoke = cv2.cvtColor(smoke, cv2.COLOR_GRAY2BGR)
            result = cv2.addWeighted(image, 1 - opacity, smoke, opacity * density, 0)
            return {"image": result}

        elif operator == "rain_effect":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for rain_effect operation")
            intensity = node.parameters.get("intensity", 0.6)
            angle = node.parameters.get("angle", 15.0)
            h, w = image.shape[:2]
            rain = np.zeros_like(image)
            num_drops = int(h * w * intensity * 0.002)
            for _ in range(num_drops):
                x = np.random.randint(0, w)
                y = np.random.randint(0, h)
                length = np.random.randint(10, 20)
                thickness = np.random.randint(1, 2)
                end_x = int(x + length * np.sin(np.deg2rad(angle)))
                end_y = int(y + length * np.cos(np.deg2rad(angle)))
                cv2.line(rain, (x, y), (end_x, end_y), (200, 200, 200), thickness)
            rain = cv2.GaussianBlur(rain, (3, 3), 0)
            result = cv2.addWeighted(image, 1, rain, intensity, 0)
            return {"image": result}

        elif operator == "snow_effect":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for snow_effect operation")
            density = node.parameters.get("density", 0.5)
            size = node.parameters.get("size", 0.8)
            h, w = image.shape[:2]
            snow = np.zeros_like(image)
            num_flakes = int(h * w * density * 0.001)
            for _ in range(num_flakes):
                x = np.random.randint(0, w)
                y = np.random.randint(0, h)
                radius = int(np.random.randint(1, 4) * size)
                cv2.circle(snow, (x, y), radius, (255, 255, 255), -1)
            snow = cv2.GaussianBlur(snow, (5, 5), 0)
            result = cv2.addWeighted(image, 1, snow, density, 0)
            return {"image": result}

        elif operator == "light_leak":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for light_leak operation")
            intensity = node.parameters.get("intensity", 0.4)
            color = np.array(node.parameters.get("color", [255, 200, 150]))
            position = node.parameters.get("position", "top")
            h, w = image.shape[:2]
            leak = np.zeros_like(image)
            if position == "top":
                leak[:h//4, :] = color
            elif position == "bottom":
                leak[-h//4:, :] = color
            elif position == "left":
                leak[:, :w//4] = color
            elif position == "right":
                leak[:, -w//4:] = color
            leak = cv2.GaussianBlur(leak, (101, 101), 0)
            result = cv2.addWeighted(image, 1, leak, intensity, 0)
            return {"image": result}

        elif operator == "film_grain":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for film_grain operation")
            intensity = node.parameters.get("intensity", 0.3)
            size = node.parameters.get("size", 1.0)
            noise = np.random.normal(0, 25 * intensity, image.shape).astype(np.float32)
            result = image.astype(np.float32) + noise
            result = np.clip(result, 0, 255).astype(np.uint8)
            return {"image": result}

        elif operator == "vignette":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for vignette operation")
            intensity = node.parameters.get("intensity", 0.5)
            radius = node.parameters.get("radius", 0.7)
            feather = node.parameters.get("feather", 0.3)
            h, w = image.shape[:2]
            Y, X = np.ogrid[:h, :w]
            center_x, center_y = w / 2, h / 2
            dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
            mask = 1 - np.clip((dist / (radius * min(center_x, center_y))) ** feather, 0, 1) * intensity
            result = image * mask[..., None]
            result = np.clip(result, 0, 255).astype(np.uint8)
            return {"image": result}

        elif operator == "lens_flare":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for lens_flare operation")
            position_x = node.parameters.get("position_x", 0.8)
            position_y = node.parameters.get("position_y", 0.2)
            intensity = node.parameters.get("intensity", 0.6)
            color = np.array(node.parameters.get("color", [255, 255, 200]))
            h, w = image.shape[:2]
            flare = np.zeros_like(image)
            cx, cy = int(position_x * w), int(position_y * h)
            cv2.circle(flare, (cx, cy), int(min(h, w) * 0.1), color.tolist(), -1)
            flare = cv2.GaussianBlur(flare, (101, 101), 0)
            result = cv2.addWeighted(image, 1, flare, intensity, 0)
            return {"image": result}

        elif operator == "chromatic_aberration":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for chromatic_aberration operation")
            red_offset = node.parameters.get("red_offset", [2, 0])
            blue_offset = node.parameters.get("blue_offset", [-2, 0])
            green_offset = node.parameters.get("green_offset", [0, 0])
            result = image.copy()
            for c, offset in zip([2, 1, 0], [red_offset, green_offset, blue_offset]):
                M = np.array([[1, 0, offset[0]], [0, 1, offset[1]]], dtype=np.float32)
                result[..., c] = cv2.warpAffine(image[..., c], M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT)
            return {"image": result}

        elif operator == "bloom":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for bloom operation")
            threshold = node.parameters.get("threshold", 200.0)
            intensity = node.parameters.get("intensity", 0.5)
            radius = node.parameters.get("radius", 10)
            mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > threshold
            bloom = cv2.GaussianBlur(image, (radius | 1, radius | 1), 0)
            result = image.copy()
            result[mask] = cv2.addWeighted(image, 1 - intensity, bloom, intensity, 0)[mask]
            return {"image": result}

        elif operator == "hdr_effect":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for hdr_effect operation")
            gamma = node.parameters.get("gamma", 0.8)
            saturation = node.parameters.get("saturation", 1.2)
            detail = node.parameters.get("detail", 0.3)
            img = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
            img = cv2.convertScaleAbs(img, alpha=saturation, beta=0)
            img = cv2.addWeighted(image, 1 - detail, img, detail, 0)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            img = cv2.LUT(img, table)
            return {"image": img}

        elif operator == "cross_processing":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for cross_processing operation")
            color_shift = node.parameters.get("color_shift", 0.3)
            contrast = node.parameters.get("contrast", 1.3)
            saturation = node.parameters.get("saturation", 1.5)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            img[..., 0] += color_shift * 30
            img[..., 1] *= saturation
            img[..., 2] *= contrast
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            return {"image": img}

        elif operator == "lomo_effect":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for lomo_effect operation")
            vignette = node.parameters.get("vignette", 0.6)
            saturation = node.parameters.get("saturation", 1.4)
            contrast = node.parameters.get("contrast", 1.2)
            light_leak = node.parameters.get("light_leak", 0.3)
            h, w = image.shape[:2]
            # Vignette
            Y, X = np.ogrid[:h, :w]
            center_x, center_y = w / 2, h / 2
            dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
            mask = 1 - np.clip((dist / (0.7 * min(center_x, center_y))) ** 2, 0, 1) * vignette
            img = image * mask[..., None]
            # Saturation/contrast
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            img[..., 1] *= saturation
            img[..., 2] *= contrast
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            # Light leak
            leak = np.zeros_like(img)
            leak[:, -w//4:] = [255, 200, 150]
            leak = cv2.GaussianBlur(leak, (101, 101), 0)
            img = cv2.addWeighted(img, 1, leak, light_leak, 0)
            return {"image": img}

        # --- New operators for advanced region processing ---
        elif operator == "mask":
            image = input_data.get("image")
            mask = input_data.get("mask")
            if image is None or mask is None:
                raise ValueError("Image and mask required for mask operation")
            # Ensure mask shape matches image
            if mask.ndim == 2 and image.ndim == 3:
                mask = mask[..., None]
            result = image * (mask > 0)
            return {"image": result}

        elif operator == "crop_region":
            image = input_data.get("image")
            if image is None:
                raise ValueError("No image provided for crop_region operation")
            x = node.parameters.get("x", 0)
            y = node.parameters.get("y", 0)
            width = node.parameters.get("width", 100)
            height = node.parameters.get("height", 100)
            result = image[y:y+height, x:x+width]
            return {"image": result}

        elif operator == "fill_region":
            image = input_data.get("image")
            mask = input_data.get("mask", None)
            color = node.parameters.get("color", [255,255,255])
            pattern = node.parameters.get("pattern", "solid")
            result = image.copy()
            if mask is not None and len(mask) > 0:
                if mask.ndim == 2 and result.ndim == 3:
                    mask = mask[..., None]
                if pattern == "solid":
                    result[mask > 0] = color
                elif pattern == "stripes":
                    # Simple horizontal stripes
                    stripe = np.zeros_like(result)
                    stripe[::10] = color
                    result[mask > 0] = stripe[mask > 0]
                elif pattern == "checkerboard":
                    cb = np.indices(result.shape[:2]).sum(axis=0) % 2
                    cb = np.repeat(cb[..., None], result.shape[2], axis=2)
                    result[(mask > 0) & (cb == 1)] = color
                # Add more patterns as needed
            else:
                if pattern == "solid":
                    result[:] = color
                elif pattern == "stripes":
                    result[::10] = color
                elif pattern == "checkerboard":
                    cb = np.indices(result.shape[:2]).sum(axis=0) % 2
                    cb = np.repeat(cb[..., None], result.shape[2], axis=2)
                    result[cb == 1] = color
            return {"image": result}

        else:
            raise ValueError(f"Unknown operator: {operator}")
    
    def execute_workflow(self, input_image: np.ndarray) -> np.ndarray:
        """Execute the complete workflow with input image"""
        if not self.nodes:
            raise ValueError("No nodes in workflow")
            
        # Determine execution order
        self.execution_order = self._determine_execution_order()
        
        # Initialize node results
        node_results = {}
        
        # Execute nodes in order
        for node_id in self.execution_order:
            node = self.nodes[node_id]
            
            # Collect inputs for this node
            input_data = {}
            for conn in self.connections:
                if conn["to_node"] == node_id:
                    from_node_id = conn["from_node"]
                    if from_node_id in node_results:
                        input_data[conn["to_input"]] = node_results[from_node_id][conn["from_output"]]
            
            # Add input image for input nodes
            if node.node_type == "input":
                input_data["image"] = input_image
            
            # Execute node
            result = self._execute_node(node, input_data)
            node_results[node_id] = result
            node.result = result
        
        # Find output node
        output_node = None
        for node in self.nodes.values():
            if node.node_type == "output":
                output_node = node
                break
                
        if output_node and output_node.node_id in node_results:
            return node_results[output_node.node_id]["image"]
        else:
            # Return result from last processing node
            for node_id in reversed(self.execution_order):
                node = self.nodes[node_id]
                if node.node_type not in ["input", "output"] and node_id in node_results:
                    return node_results[node_id]["image"]
                    
        raise ValueError("No output found in workflow")

class AIWorkflowGenerator:
    """AI-powered workflow generator using OpenAI API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key
        self.processor = WorkflowProcessor()
        
    def generate_workflow_from_prompt(self, image: np.ndarray, prompt: str) -> Dict[str, Any]:
        """Generate a workflow from a natural language prompt"""
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
            
        # Convert image to base64 for API
        image_base64 = self._image_to_base64(image)
        
        # Read node specifications
        with open('node_spec.txt', 'r') as f:
            node_specs = f.read()
        
        # Create system prompt
        system_prompt = f"""You are an expert in computer vision and image processing. You need to create a workflow of image processing operations based on a user's request.

Available node specifications:
{node_specs}

Your task is to:
1. Analyze the user's request
2. Create a JSON workflow that describes the nodes and their connections
3. Choose appropriate parameters for each node
4. Ensure the workflow is valid and executable

The workflow should be returned as a JSON object with this structure:
{{
    "nodes": [
        {{
            "node_id": "unique_id",
            "node_type": "node_name",
            "parameters": {{}},
            "inputs": [],
            "outputs": []
        }}
    ],
    "connections": [
        {{
            "from_node": "node_id",
            "to_node": "node_id",
            "from_output": "image",
            "to_input": "image"
        }}
    ]
}}

CRITICAL RULES:
- Always start with an "input" node
- Always end with an "output" node
- Connect nodes sequentially through their inputs/outputs
- Use appropriate parameters for each operation
- Keep the workflow simple and focused on the user's request
- Ensure all connections are valid (image to image)
- Use ONLY valid JSON syntax - no mathematical expressions like "1280 - 300"
- Use actual calculated values instead of expressions
- For positioning, use reasonable fixed values (e.g., 50, 100, 200)
- For colors, use BGR format as arrays: [255, 255, 255] for white, [0, 0, 255] for red
- For text positioning, use simple integers like 50, 100, 200
- Do not use variables or calculations in JSON values
- **If you use a 'mask' node, you MUST always create a node that produces a mask (such as a threshold, segmentation, or similar node) and connect its output to the 'mask' input of the 'mask' node. The 'mask' input must never be missing.**"""
        
        # Create user prompt
        user_prompt = f"""Please create an image processing workflow for the following request:

User Request: {prompt}

The input image has dimensions: {image.shape[1]}x{image.shape[0]} pixels
Image type: {'Grayscale' if len(image.shape) == 2 else 'Color'} ({image.shape[2] if len(image.shape) == 3 else 1} channels)

Please generate a JSON workflow that will process this image according to the user's request."""
        
        try:
            # Call OpenAI API
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            # Extract and parse JSON from response
            response_text = response.choices[0].message.content

            print(response_text)

            if response_text is None:
                raise ValueError("No content received from OpenAI API")
            
            print("Extracting JSON from response...")
            workflow_json = self._extract_json_from_response(response_text)
            
            print(f"Generated workflow JSON: {workflow_json}")
            
            return workflow_json
            
        except Exception as e:
            print(f"Failed to generate workflow: {str(e)}")
            print("Creating fallback workflow...")
            return self._create_fallback_workflow(prompt)
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        import cv2
        """Convert numpy image to base64 string"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Convert to PIL Image and then to base64
        pil_image = Image.fromarray(image_rgb)
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from OpenAI response"""
        # Try to find JSON in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON found in response")
            
        json_str = response_text[json_start:json_end]
        
        # Clean up common JSON issues
        json_str = self._clean_json_string(json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            json_str = self._fix_json_issues(json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e2:
                raise ValueError(f"Invalid JSON in response after fixing: {e2}. Original error: {e}")
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean up common JSON string issues"""
        # Remove any trailing commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Remove any comments (though JSON doesn't support comments)
        json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        return json_str
    
    def _fix_json_issues(self, json_str: str) -> str:
        """Fix common JSON issues like mathematical expressions"""
        # Fix mathematical expressions in values
        # Replace expressions like "1280 - 300" with calculated values
        def replace_math(match):
            try:
                # Extract the mathematical expression
                expr = match.group(1)
                # Evaluate the expression safely
                result = eval(expr)
                return str(result)
            except:
                # If evaluation fails, return a default value
                return "100"
        
        # Find and replace mathematical expressions in parameter values
        json_str = re.sub(r':\s*"([^"]*)"', lambda m: self._fix_math_in_string(m), json_str)
        json_str = re.sub(r':\s*(\d+\s*[-+*/]\s*\d+)', replace_math, json_str)
        
        return json_str
    
    def _fix_math_in_string(self, match) -> str:
        """Fix mathematical expressions within quoted strings"""
        content = match.group(1)
        # Look for patterns like "1280 - 300" within the string
        if re.search(r'\d+\s*[-+*/]\s*\d+', content):
            try:
                # Extract and evaluate the mathematical expression
                expr_match = re.search(r'(\d+\s*[-+*/]\s*\d+)', content)
                if expr_match:
                    expr = expr_match.group(1)
                    result = eval(expr)
                    # Replace the expression with the result
                    content = content.replace(expr, str(result))
            except:
                # If evaluation fails, keep the original
                pass
        return f': "{content}"'
    
    def create_workflow_from_json(self, workflow_json: Dict[str, Any]) -> WorkflowProcessor:
        """Create a WorkflowProcessor instance from JSON workflow"""
        processor = WorkflowProcessor()
        
        # Add nodes
        for node_data in workflow_json.get("nodes", []):
            node = Node.from_dict(node_data)
            processor.add_node(node)
        
        # Add connections
        for conn_data in workflow_json.get("connections", []):
            processor.add_connection(
                conn_data["from_node"],
                conn_data["to_node"],
                conn_data.get("from_output", "image"),
                conn_data.get("to_input", "image")
            )
        
        return processor
    
    def _create_fallback_workflow(self, prompt: str) -> Dict[str, Any]:
        """Create a simple fallback workflow when AI generation fails"""
        # Create a basic workflow based on common keywords in the prompt
        prompt_lower = prompt.lower()
        
        nodes = [
            {
                "node_id": "input_node",
                "node_type": "input",
                "parameters": {},
                "inputs": [],
                "outputs": ["image"]
            }
        ]
        
        # Add processing nodes based on prompt keywords
        if "gray" in prompt_lower or "grayscale" in prompt_lower:
            nodes.append({
                "node_id": "gray_node",
                "node_type": "rgb2gray",
                "parameters": {},
                "inputs": ["image"],
                "outputs": ["image"]
            })
        
        if "blur" in prompt_lower:
            nodes.append({
                "node_id": "blur_node",
                "node_type": "gaussian_blur",
                "parameters": {"kernel_size": [5, 5]},
                "inputs": ["image"],
                "outputs": ["image"]
            })
        
        if "text" in prompt_lower:
            nodes.append({
                "node_id": "text_node",
                "node_type": "add_text",
                "parameters": {
                    "text": "Sample Text",
                    "x": 50,
                    "y": 50,
                    "font_scale": 1.0,
                    "color": [255, 255, 255],
                    "thickness": 2,
                    "font": "FONT_HERSHEY_SIMPLEX"
                },
                "inputs": ["image"],
                "outputs": ["image"]
            })
        
        # Add output node
        nodes.append({
            "node_id": "output_node",
            "node_type": "output",
            "parameters": {},
            "inputs": ["image"],
            "outputs": []
        })
        
        # Create connections
        connections = []
        for i in range(len(nodes) - 1):
            connections.append({
                "from_node": nodes[i]["node_id"],
                "to_node": nodes[i + 1]["node_id"],
                "from_output": "image",
                "to_input": "image"
            })
        
        return {
            "nodes": nodes,
            "connections": connections
        }

def process_image_with_ai_workflow(image: np.ndarray, prompt: str, api_key: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Main function to process image with AI-generated workflow"""
    generator = AIWorkflowGenerator(api_key)
    
    # Generate workflow
    workflow_json = generator.generate_workflow_from_prompt(image, prompt)
    
    # Create and execute workflow
    processor = generator.create_workflow_from_json(workflow_json)
    result_image = processor.execute_workflow(image)
    
    return result_image, workflow_json 