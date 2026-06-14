"""
4_claude_agent.py — Claude agent for computational lithography

The agent receives a layout and decides which tools to call:
1. run_opc_correction(layout) → corrected mask
2. check_hotspots(layout) → risk heatmap
3. generate_report(data) → natural language analysis

Agent reasoning: Analyzes the pattern type, decides which analyses are needed,
interprets results, generates a detailed technical report.

Usage:
    from claude_agent import analyze_layout
    
    result = analyze_layout(
        layout_image=layout_array,  # numpy array (256, 256)
        api_key='sk-...'
    )
    print(result['report'])
"""

import anthropic
import base64
import json
import numpy as np
import io
from PIL import Image
from typing import Any
import torch
import cv2
from pathlib import Path


# Load models (global, loaded once)
OPC_MODEL = None
HOTSPOT_MODEL = None
DEVICE = None


def init_models(device='cpu'):
    """Initialize ML models"""
    global OPC_MODEL, HOTSPOT_MODEL, DEVICE
    
    DEVICE = torch.device(device)
    
    # Import here to avoid requiring torch if not using
    from train_opc_model import SimpleUNet as OPCNet
    from train_hotspot_model import SimpleHotspotCNN
    
    # Load OPC model
    opc_path = Path('models/opc_unet.pth')
    if opc_path.exists():
        OPC_MODEL = OPCNet(in_channels=1, out_channels=1)
        OPC_MODEL.load_state_dict(torch.load(opc_path, map_location=DEVICE))
        OPC_MODEL.eval()
    
    # Load hotspot model
    hotspot_path = Path('models/hotspot_cnn.pth')
    if hotspot_path.exists():
        HOTSPOT_MODEL = SimpleHotspotCNN(in_channels=1)
        HOTSPOT_MODEL.load_state_dict(torch.load(hotspot_path, map_location=DEVICE))
        HOTSPOT_MODEL.eval()


def run_opc_correction(layout_base64: str) -> dict:
    """
    Tool: Run OPC U-Net model to correct layout
    Input: base64-encoded layout image
    Output: base64-encoded corrected mask + metrics
    """
    if OPC_MODEL is None:
        return {
            'success': False,
            'error': 'OPC model not loaded',
            'corrected_mask': None
        }
    
    # Decode image
    layout_data = base64.b64decode(layout_base64)
    layout_img = Image.open(io.BytesIO(layout_data)).convert('L')
    layout_arr = np.array(layout_img, dtype=np.float32) / 255.0
    
    # Prepare input
    layout_tensor = torch.from_numpy(layout_arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 256, 256)
    layout_tensor = layout_tensor.to(DEVICE)
    
    # Run model
    with torch.no_grad():
        corrected = OPC_MODEL(layout_tensor)
    
    corrected_np = corrected[0, 0].cpu().numpy()
    corrected_np = (corrected_np * 255).astype(np.uint8)
    
    # Encode result
    _, buffer = cv2.imencode('.png', corrected_np)
    corrected_b64 = base64.b64encode(buffer).decode()
    
    # Compute metrics
    edge_shift = float(np.abs(layout_arr - corrected_np/255.0).mean()) * 100
    
    return {
        'success': True,
        'corrected_mask': corrected_b64,
        'edge_shift_nanometers': round(edge_shift * 2.1, 2),
        'note': 'Model predicted mask corrections for proximity and diffraction effects'
    }


def check_hotspots(layout_base64: str) -> dict:
    """
    Tool: Run hotspot detector to identify high-risk regions
    Input: base64-encoded layout
    Output: risk heatmap + risk scores
    """
    if HOTSPOT_MODEL is None:
        return {
            'success': False,
            'error': 'Hotspot model not loaded',
            'heatmap': None
        }
    
    # Decode image
    layout_data = base64.b64decode(layout_base64)
    layout_img = Image.open(io.BytesIO(layout_data)).convert('L')
    layout_arr = np.array(layout_img, dtype=np.float32) / 255.0
    
    # Extract patches and evaluate
    patch_size = 64
    stride = 32
    h, w = layout_arr.shape
    
    risk_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    
    patches_risks = []
    
    with torch.no_grad():
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = layout_arr[y:y+patch_size, x:x+patch_size]
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)
                
                risk_score = float(HOTSPOT_MODEL(patch_tensor)[0].cpu().numpy())
                risk_map[y:y+patch_size, x:x+patch_size] += risk_score
                count_map[y:y+patch_size, x:x+patch_size] += 1
                
                patches_risks.append({
                    'position': (int(y), int(x)),
                    'risk': round(risk_score, 3)
                })
    
    # Average overlapping regions
    risk_map = np.divide(risk_map, count_map, where=count_map>0, out=risk_map)
    
    # Encode heatmap
    risk_visual = (risk_map * 255).astype(np.uint8)
    risk_visual = cv2.applyColorMap(risk_visual, cv2.COLORMAP_JET)
    _, buffer = cv2.imencode('.png', risk_visual)
    heatmap_b64 = base64.b64encode(buffer).decode()
    
    # Compute stats
    high_risk = np.sum(risk_map > 0.7)
    med_risk = np.sum((risk_map > 0.4) & (risk_map <= 0.7))
    low_risk = np.sum(risk_map <= 0.4)
    
    return {
        'success': True,
        'heatmap': heatmap_b64,
        'high_risk_regions': int(high_risk),
        'medium_risk_regions': int(med_risk),
        'low_risk_regions': int(low_risk),
        'average_risk': round(float(np.mean(risk_map)), 3),
        'max_risk': round(float(np.max(risk_map)), 3)
    }


def analyze_layout(layout_image: np.ndarray, api_key: str = None) -> dict:
    """
    Main agent interface.
    
    Args:
        layout_image: numpy array (256, 256), uint8 or float [0,255] or [0,1]
        api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
    
    Returns:
        dict with keys:
        - report: natural language analysis
        - corrected_mask: base64 image
        - hotspot_heatmap: base64 image
        - metrics: dict of computed metrics
        - reasoning: agent's reasoning steps
    """
    
    # Normalize image
    if layout_image.dtype == np.float32 or layout_image.dtype == np.float64:
        layout_image = (layout_image * 255).astype(np.uint8)
    
    # Encode layout for Claude
    layout_pil = Image.fromarray(layout_image, mode='L')
    layout_buffer = io.BytesIO()
    layout_pil.save(layout_buffer, format='PNG')
    layout_b64 = base64.b64encode(layout_buffer.getvalue()).decode()
    
    # Initialize client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Define tools
    tools = [
        {
            "name": "run_opc_correction",
            "description": "Run the OPC (Optical Proximity Correction) U-Net model to compute mask corrections for diffraction effects",
            "input_schema": {
                "type": "object",
                "properties": {
                    "layout_base64": {
                        "type": "string",
                        "description": "Base64-encoded layout image"
                    }
                },
                "required": ["layout_base64"]
            }
        },
        {
            "name": "check_hotspots",
            "description": "Run the CNN hotspot detector to identify high-risk regions with high edge placement error",
            "input_schema": {
                "type": "object",
                "properties": {
                    "layout_base64": {
                        "type": "string",
                        "description": "Base64-encoded layout image"
                    }
                },
                "required": ["layout_base64"]
            }
        }
    ]
    
    # Agent loop
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": layout_b64
                    }
                },
                {
                    "type": "text",
                    "text": """Analyze this semiconductor layout pattern for computational lithography.

You are an expert lithography engineer using ML-enhanced tools. Your job:
1. Examine the layout pattern
2. Identify pattern type and characteristics
3. Run OPC correction to predict how light diffraction will affect it
4. Check for hotspots (regions likely to fail manufacturing)
5. Generate a technical report with recommendations

Use the tools to analyze the layout thoroughly. Then provide a report in plain English suitable for a chip designer."""
                }
            ]
        }
    ]
    
    reasoning_steps = []
    all_results = {}
    
    # Agentic loop
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            tools=tools,
            messages=messages
        )
        
        # Check stop reason
        if response.stop_reason == "end_turn":
            # Agent finished reasoning
            final_text = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    final_text = block.text
                    break
            break
        
        # Process tool calls
        if response.stop_reason == "tool_use":
            assistant_message = {"role": "assistant", "content": response.content}
            messages.append(assistant_message)
            
            tool_results = []
            
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    
                    # Execute tool
                    if tool_name == "run_opc_correction":
                        result = run_opc_correction(layout_b64)
                        all_results['opc'] = result
                        reasoning_steps.append(f"Ran OPC correction: {result.get('note', 'Success')}")
                    
                    elif tool_name == "check_hotspots":
                        result = check_hotspots(layout_b64)
                        all_results['hotspots'] = result
                        reasoning_steps.append(
                            f"Detected hotspots: {result['high_risk_regions']} high-risk, "
                            f"{result['medium_risk_regions']} medium-risk regions"
                        )
                    
                    else:
                        result = {"error": f"Unknown tool: {tool_name}"}
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })
            
            # Add tool results to conversation
            messages.append({
                "role": "user",
                "content": tool_results
            })
        else:
            # Unexpected stop reason
            break
    
    # Compile final result
    return {
        'report': final_text if iteration < max_iterations else "Agent did not complete analysis.",
        'corrected_mask': all_results.get('opc', {}).get('corrected_mask'),
        'hotspot_heatmap': all_results.get('hotspots', {}).get('heatmap'),
        'metrics': {
            'opc': all_results.get('opc', {}),
            'hotspots': {k: v for k, v in all_results.get('hotspots', {}).items() 
                        if k not in ['success', 'heatmap']}
        },
        'reasoning': reasoning_steps,
        'agent_iterations': iteration
    }


if __name__ == '__main__':
    # Example usage
    init_models(device='cpu')
    
    # Create a dummy layout for testing
    dummy_layout = np.zeros((256, 256), dtype=np.uint8)
    dummy_layout[50:100, 50:200] = 255
    dummy_layout[150:170, 100:250] = 255
    
    result = analyze_layout(dummy_layout)
    print("\n=== Agent Analysis ===")
    print("\nReport:")
    print(result['report'])
    print("\nReasoning steps:")
    for step in result['reasoning']:
        print(f"  • {step}")
    print(f"\nAgent iterations: {result['agent_iterations']}")
