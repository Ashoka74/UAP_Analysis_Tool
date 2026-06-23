from runpod_flash import Endpoint, GpuType
import asyncio

@Endpoint(name="flash-quickstart", gpu=GpuType.NVIDIA_GEFORCE_RTX_4090, dependencies=["torch"])
def gpu_compute(data):
  import torch
  tensor = torch.tensor(data, device="cuda")
  return {"result": tensor.sum().item(), "device": torch.cuda.get_device_name(0)}

async def main():
  result = await gpu_compute([1, 2, 3, 4, 5])
  print(f"Sum: {result['result']}\nComputed on: {result['device']}")

if __name__ == "__main__":
  asyncio.run(main())
