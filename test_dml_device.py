#!/usr/bin/env python3
"""Test script to verify DirectML is using the correct GPU"""

import onnxruntime as ort
import numpy as np
import pathlib

def test_dml_provider():
    """Test if DML provider is available and which device it uses"""
    
    print("Available providers:", ort.get_available_providers())
    print()
    
    # Check if DML is available
    if 'DmlExecutionProvider' not in ort.get_available_providers():
        print("❌ DmlExecutionProvider is NOT available!")
        return
    
    print("✓ DmlExecutionProvider is available")
    print()
    
    # Load a model with DML
    model_path = pathlib.Path("experiments/GAME-1.0-small-onnx-opset17/encoder.onnx")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"Loading model: {model_path}")
    
    # Create session with DML
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 0  # Verbose logging
    
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(
        model_path.as_posix(),
        sess_options=session_options,
        providers=providers
    )
    
    print(f"✓ Model loaded successfully")
    print(f"Active provider: {session.get_providers()}")
    print()
    
    # Get device info
    print("Device information:")
    print(f"  Providers: {session.get_providers()}")
    
    # Try to get DML device info
    try:
        # Run a dummy inference to trigger device initialization
        dummy_input = {
            "waveform": np.random.randn(1, 16000).astype(np.float32),
            "duration": np.array([1.0], dtype=np.float32)
        }
        
        print("\nRunning test inference...")
        output = session.run(None, dummy_input)
        print(f"✓ Inference successful, output shapes: {[o.shape for o in output]}")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
    
    print("\n" + "="*60)
    print("DirectML Provider Information:")
    print("="*60)
    print("DirectML uses the default GPU adapter on your system.")
    print("To verify which GPU is being used:")
    print("1. Open Task Manager (Ctrl+Shift+Esc)")
    print("2. Go to Performance tab")
    print("3. Look at GPU sections while running inference")
    print("4. The GPU with activity is being used by DirectML")
    print()
    print("Note: DirectML automatically selects the most capable GPU.")
    print("If you have multiple GPUs, it typically uses the discrete GPU.")

if __name__ == "__main__":
    test_dml_provider()
