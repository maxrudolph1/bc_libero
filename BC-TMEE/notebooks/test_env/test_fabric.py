import torch
from lightning.fabric import Fabric


def diagnose_fabric_setup():
    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("Warning: CUDA is not available")
            return

        # Check if the number of devices matches
        cuda_device_count = torch.cuda.device_count()
        config_gpu_count = len([0])
        
        if config_gpu_count > cuda_device_count:
            print(f"Warning: Configured GPU count ({config_gpu_count}) exceeds the available GPU count ({cuda_device_count})")
            return

        # Attempt to create a Fabric instance
        fabric = Fabric(
            accelerator="cuda", 
            devices=[0], 
            precision="bf16-mixed", 
            strategy="deepspeed"
        )
        print("Fabric instance created successfully")

    except Exception as e:
        print(f"Error occurred while creating Fabric instance: {e}")
        import traceback
        traceback.print_exc()

# Call the diagnostic function
diagnose_fabric_setup()
